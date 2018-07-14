use core::f64;

const TINY: f64 = 1.0e-300;

#[inline]
pub fn sqrt(x: f64) -> f64 {
    let mut z: f64;
    let sign: u32 = 0x80000000;
    let mut ix0: i32;
    let mut s0: i32;
    let mut q: i32;
    let mut m: i32;
    let mut t: i32;
    let mut i: i32;
    let mut r: u32;
    let mut t1: u32;
    let mut s1: u32;
    let mut ix1: u32;
    let mut q1: u32;

    ix0 = (x.to_bits() >> 32) as i32;
    ix1 = x.to_bits() as u32;

    /* take care of Inf and NaN */
    if (ix0&0x7ff00000) == 0x7ff00000 {
        return x*x + x;  /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
    }
    /* take care of zero */
    if ix0 <= 0 {
        if ((ix0&!(sign as i32))|ix1 as i32) == 0 {
            return x;  /* sqrt(+-0) = +-0 */
        }
        if ix0 < 0 {
            return (x - x) / (x - x);  /* sqrt(-ve) = sNaN */
        }
    }
    /* normalize x */
    m = ix0>>20;
    if m == 0 {  /* subnormal x */
        while ix0 == 0 {
            m -= 21;
            ix0 |= (ix1>>11) as i32;
            ix1 <<= 21;
        }
        i=0;
        while (ix0&0x00100000) == 0 {
            i += 1;
            ix0 <<= 1;
        }
        m -= i - 1;
        ix0 |= (ix1>>(32-i)) as i32;
        ix1 <<= i;
    }
    m -= 1023;    /* unbias exponent */
    ix0 = (ix0&0x000fffff)|0x00100000;
    if (m & 1) == 1 {  /* odd m, double x to make it even */
        ix0 += ix0 + ((ix1&sign)>>31) as i32;
        ix1 += ix1;
    }
    m >>= 1;      /* m = [m/2] */

    /* generate sqrt(x) bit by bit */
    ix0 += ix0 + ((ix1&sign)>>31) as i32;
    ix1 += ix1;
    q = 0; /* [q,q1] = sqrt(x) */
    q1 = 0;
    s0 = 0;
    s1 = 0;
    r = 0x00200000;        /* r = moving bit from right to left */

    while r != 0 {
        t = s0 + r as i32;
        if t <= ix0 {
            s0   = t + r as i32;
            ix0 -= t;
            q   += r as i32;
        }
        ix0 += ix0 + ((ix1&sign)>>31) as i32;
        ix1 += ix1;
        r >>= 1;
    }

    r = sign;
    while r != 0 {
        t1 = s1 + r;
        t  = s0;
        if t < ix0 || (t == ix0 && t1 <= ix1) {
            s1 = t1 + r;
            if (t1&sign) == sign && (s1&sign) == 0 {
                s0 += 1;
            }
            ix0 -= t;
            if ix1 < t1 {
                ix0 -= 1;
            }
            ix1 -= t1;
            q1 += r;
        }
        ix0 += ix0 + ((ix1&sign)>>31) as i32;
        ix1 += ix1;
        r >>= 1;
    }

    /* use floating add to find out rounding direction */
    if (ix0 as u32|ix1) != 0 {
        z = 1.0 - TINY; /* raise inexact flag */
        if z >= 1.0 {
            z = 1.0 + TINY;
            if q1 == 0xffffffff {
                q1 = 0;
                q+=1;
            } else if z > 1.0 {
                if q1 == 0xfffffffe {
                    q += 1;
                }
                q1 += 2;
            } else {
                q1 += q1 & 1;
            }
        }
    }
    ix0 = (q>>1) + 0x3fe00000;
    ix1 = q1>>1;
    if (q&1) == 1 {
        ix1 |= sign;
    }
    ix0 += m << 20;
    f64::from_bits((ix0 as u64) << 32 | ix1 as u64)
}
