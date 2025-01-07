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

/// The square root of `x` (f32).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn sqrtf(x: f32) -> f32 {
    select_implementation! {
        name: sqrtf,
        use_arch: any(
            all(target_arch = "wasm32", intrinsics_enabled),
            target_feature = "sse2"
        ),
        args: x,
    }

    const TINY: f32 = 1.0e-30;

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

// PowerPC tests are failing on LLVM 13: https://github.com/rust-lang/rust/issues/88520
#[cfg(not(target_arch = "powerpc64"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_check() {
        assert_eq!(sqrtf(100.0), 10.0);
        assert_eq!(sqrtf(4.0), 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/sqrt
    #[test]
    fn spec_tests() {
        // Not Asserted: FE_INVALID exception is raised if argument is negative.
        assert!(sqrtf(-1.0).is_nan());
        assert!(sqrtf(f32::NAN).is_nan());
        for f in [0.0, -0.0, f32::INFINITY].iter().copied() {
            assert_eq!(sqrtf(f), f);
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn conformance_tests() {
        let values = [3.14159265359f32, 10000.0f32, f32::from_bits(0x0000000f), f32::INFINITY];
        let results = [1071833029u32, 1120403456u32, 456082799u32, 2139095040u32];

        for i in 0..values.len() {
            let bits = f32::to_bits(sqrtf(values[i]));
            assert_eq!(results[i], bits);
        }
    }
}
