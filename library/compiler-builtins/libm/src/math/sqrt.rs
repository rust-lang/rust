/* origin: FreeBSD /usr/src/lib/msun/src/e_sqrt.c */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* sqrt(x)
 * Return correctly rounded sqrt.
 *           ------------------------------------------
 *           |  Use the hardware sqrt if you have one |
 *           ------------------------------------------
 * Method:
 *   Bit by bit method using integer arithmetic. (Slow, but portable)
 *   1. Normalization
 *      Scale x to y in [1,4) with even powers of 2:
 *      find an integer k such that  1 <= (y=x*2^(2k)) < 4, then
 *              sqrt(x) = 2^k * sqrt(y)
 *   2. Bit by bit computation
 *      Let q  = sqrt(y) truncated to i bit after binary point (q = 1),
 *           i                                                   0
 *                                     i+1         2
 *          s  = 2*q , and      y  =  2   * ( y - q  ).         (1)
 *           i      i            i                 i
 *
 *      To compute q    from q , one checks whether
 *                  i+1       i
 *
 *                            -(i+1) 2
 *                      (q + 2      ) <= y.                     (2)
 *                        i
 *                                                            -(i+1)
 *      If (2) is false, then q   = q ; otherwise q   = q  + 2      .
 *                             i+1   i             i+1   i
 *
 *      With some algebraic manipulation, it is not difficult to see
 *      that (2) is equivalent to
 *                             -(i+1)
 *                      s  +  2       <= y                      (3)
 *                       i                i
 *
 *      The advantage of (3) is that s  and y  can be computed by
 *                                    i      i
 *      the following recurrence formula:
 *          if (3) is false
 *
 *          s     =  s  ,       y    = y   ;                    (4)
 *           i+1      i          i+1    i
 *
 *          otherwise,
 *                         -i                     -(i+1)
 *          s     =  s  + 2  ,  y    = y  -  s  - 2             (5)
 *           i+1      i          i+1    i     i
 *
 *      One may easily use induction to prove (4) and (5).
 *      Note. Since the left hand side of (3) contain only i+2 bits,
 *            it does not necessary to do a full (53-bit) comparison
 *            in (3).
 *   3. Final rounding
 *      After generating the 53 bits result, we compute one more bit.
 *      Together with the remainder, we can decide whether the
 *      result is exact, bigger than 1/2ulp, or less than 1/2ulp
 *      (it will never equal to 1/2ulp).
 *      The rounding mode can be detected by checking whether
 *      huge + tiny is equal to huge, and whether huge - tiny is
 *      equal to huge for some floating point number "huge" and "tiny".
 *
 * Special cases:
 *      sqrt(+-0) = +-0         ... exact
 *      sqrt(inf) = inf
 *      sqrt(-ve) = NaN         ... with invalid signal
 *      sqrt(NaN) = NaN         ... with invalid signal for signaling NaN
 */

use core::f64;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn sqrt(x: f64) -> f64 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f64.sqrt` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32")] {
            return if x < 0.0 {
                f64::NAN
            } else {
                unsafe { ::core::intrinsics::sqrtf64(x) }
            }
        }
    }
    #[cfg(target_feature = "sse2")]
    {
        // Note: This path is unlikely since LLVM will usually have already
        // optimized sqrt calls into hardware instructions if sse2 is available,
        // but if someone does end up here they'll apprected the speed increase.
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        unsafe {
            let m = _mm_set_sd(x);
            let m_sqrt = _mm_sqrt_pd(m);
            _mm_cvtsd_f64(m_sqrt)
        }
    }
    #[cfg(not(target_feature = "sse2"))]
    {
        use core::num::Wrapping;

        const TINY: f64 = 1.0e-300;

        let mut z: f64;
        let sign: Wrapping<u32> = Wrapping(0x80000000);
        let mut ix0: i32;
        let mut s0: i32;
        let mut q: i32;
        let mut m: i32;
        let mut t: i32;
        let mut i: i32;
        let mut r: Wrapping<u32>;
        let mut t1: Wrapping<u32>;
        let mut s1: Wrapping<u32>;
        let mut ix1: Wrapping<u32>;
        let mut q1: Wrapping<u32>;

        ix0 = (x.to_bits() >> 32) as i32;
        ix1 = Wrapping(x.to_bits() as u32);

        /* take care of Inf and NaN */
        if (ix0 & 0x7ff00000) == 0x7ff00000 {
            return x * x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
        }
        /* take care of zero */
        if ix0 <= 0 {
            if ((ix0 & !(sign.0 as i32)) | ix1.0 as i32) == 0 {
                return x; /* sqrt(+-0) = +-0 */
            }
            if ix0 < 0 {
                return (x - x) / (x - x); /* sqrt(-ve) = sNaN */
            }
        }
        /* normalize x */
        m = ix0 >> 20;
        if m == 0 {
            /* subnormal x */
            while ix0 == 0 {
                m -= 21;
                ix0 |= (ix1 >> 11).0 as i32;
                ix1 <<= 21;
            }
            i = 0;
            while (ix0 & 0x00100000) == 0 {
                i += 1;
                ix0 <<= 1;
            }
            m -= i - 1;
            ix0 |= (ix1 >> (32 - i) as usize).0 as i32;
            ix1 = ix1 << i as usize;
        }
        m -= 1023; /* unbias exponent */
        ix0 = (ix0 & 0x000fffff) | 0x00100000;
        if (m & 1) == 1 {
            /* odd m, double x to make it even */
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
        }
        m >>= 1; /* m = [m/2] */

        /* generate sqrt(x) bit by bit */
        ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
        ix1 += ix1;
        q = 0; /* [q,q1] = sqrt(x) */
        q1 = Wrapping(0);
        s0 = 0;
        s1 = Wrapping(0);
        r = Wrapping(0x00200000); /* r = moving bit from right to left */

        while r != Wrapping(0) {
            t = s0 + r.0 as i32;
            if t <= ix0 {
                s0 = t + r.0 as i32;
                ix0 -= t;
                q += r.0 as i32;
            }
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
            r >>= 1;
        }

        r = sign;
        while r != Wrapping(0) {
            t1 = s1 + r;
            t = s0;
            if t < ix0 || (t == ix0 && t1 <= ix1) {
                s1 = t1 + r;
                if (t1 & sign) == sign && (s1 & sign) == Wrapping(0) {
                    s0 += 1;
                }
                ix0 -= t;
                if ix1 < t1 {
                    ix0 -= 1;
                }
                ix1 -= t1;
                q1 += r;
            }
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
            r >>= 1;
        }

        /* use floating add to find out rounding direction */
        if (ix0 as u32 | ix1.0) != 0 {
            z = 1.0 - TINY; /* raise inexact flag */
            if z >= 1.0 {
                z = 1.0 + TINY;
                if q1.0 == 0xffffffff {
                    q1 = Wrapping(0);
                    q += 1;
                } else if z > 1.0 {
                    if q1.0 == 0xfffffffe {
                        q += 1;
                    }
                    q1 += Wrapping(2);
                } else {
                    q1 += q1 & Wrapping(1);
                }
            }
        }
        ix0 = (q >> 1) + 0x3fe00000;
        ix1 = q1 >> 1;
        if (q & 1) == 1 {
            ix1 |= sign;
        }
        ix0 += m << 20;
        f64::from_bits((ix0 as u64) << 32 | ix1.0 as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::*;

    #[test]
    fn sanity_check() {
        assert_eq!(sqrt(100.0), 10.0);
        assert_eq!(sqrt(4.0), 2.0);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/sqrt
    #[test]
    fn spec_tests() {
        // Not Asserted: FE_INVALID exception is raised if argument is negative.
        assert!(sqrt(-1.0).is_nan());
        assert!(sqrt(NAN).is_nan());
        for f in [0.0, -0.0, INFINITY].iter().copied() {
            assert_eq!(sqrt(f), f);
        }
    }
    
    #[test]
    fn conformance_tests() {
        let values = [3.14159265359, 10000.0, f64::from_bits(0x0000000f), INFINITY];
        let results = [4610661241675116657u64, 4636737291354636288u64,
        2197470602079456986u64, 9218868437227405312u64];
        
        for i in 0..values.len() {
            let bits = f64::to_bits(sqrt(values[i]));
            assert_eq!(results[i], bits);
        } 
    }
}
