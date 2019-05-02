/* origin: FreeBSD /usr/src/lib/msun/src/e_sqrtf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
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

const TINY: f32 = 1.0e-30;

#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn sqrtf(x: f32) -> f32 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f32.sqrt` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32")] {
            return if x < 0.0 {
                ::core::f32::NAN
            } else {
                unsafe { ::core::intrinsics::sqrtf32(x) }
            }
        }
    }
    let mut z: f32;
    let sign: i32 = 0x80000000u32 as i32;
    let mut ix: i32;
    let mut s: i32;
    let mut q: i32;
    let mut m: i32;
    let mut t: i32;
    let mut i: i32;
    let mut r: u32;

    ix = x.to_bits() as i32;

    /* take care of Inf and NaN */
    if (ix as u32 & 0x7f800000) == 0x7f800000 {
        return x * x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
    }

    /* take care of zero */
    if ix <= 0 {
        if (ix & !sign) == 0 {
            return x; /* sqrt(+-0) = +-0 */
        }
        if ix < 0 {
            return (x - x) / (x - x); /* sqrt(-ve) = sNaN */
        }
    }

    /* normalize x */
    m = ix >> 23;
    if m == 0 {
        /* subnormal x */
        i = 0;
        while ix & 0x00800000 == 0 {
            ix <<= 1;
            i = i + 1;
        }
        m -= i - 1;
    }
    m -= 127; /* unbias exponent */
    ix = (ix & 0x007fffff) | 0x00800000;
    if m & 1 == 1 {
        /* odd m, double x to make it even */
        ix += ix;
    }
    m >>= 1; /* m = [m/2] */

    /* generate sqrt(x) bit by bit */
    ix += ix;
    q = 0;
    s = 0;
    r = 0x01000000; /* r = moving bit from right to left */

    while r != 0 {
        t = s + r as i32;
        if t <= ix {
            s = t + r as i32;
            ix -= t;
            q += r as i32;
        }
        ix += ix;
        r >>= 1;
    }

    /* use floating add to find out rounding direction */
    if ix != 0 {
        z = 1.0 - TINY; /* raise inexact flag */
        if z >= 1.0 {
            z = 1.0 + TINY;
            if z > 1.0 {
                q += 2;
            } else {
                q += q & 1;
            }
        }
    }

    ix = (q >> 1) + 0x3f000000;
    ix += m << 23;
    f32::from_bits(ix as u32)
}
