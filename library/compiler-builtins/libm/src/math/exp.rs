use super::scalbn;

const HALF: [f64; 2] = [0.5, -0.5];
const LN2_HI: f64 = 6.93147180369123816490e-01; /* 0x3fe62e42, 0xfee00000 */
const LN2_LO: f64 = 1.90821492927058770002e-10; /* 0x3dea39ef, 0x35793c76 */
const INV_LN2: f64 = 1.44269504088896338700e+00; /* 0x3ff71547, 0x652b82fe */
const P1: f64 = 1.66666666666666019037e-01; /* 0x3FC55555, 0x5555553E */
const P2: f64 = -2.77777777770155933842e-03; /* 0xBF66C16C, 0x16BEBD93 */
const P3: f64 = 6.61375632143793436117e-05; /* 0x3F11566A, 0xAF25DE2C */
const P4: f64 = -1.65339022054652515390e-06; /* 0xBEBBBD41, 0xC5D26BF1 */
const P5: f64 = 4.13813679705723846039e-08; /* 0x3E663769, 0x72BEA4D0 */

#[inline]
pub fn exp(mut x: f64) -> f64 {
    let mut hx: u32 = (x.to_bits() >> 32) as u32;
    let sign = (hx >> 31) as i32; /* sign bit of x */
    hx &= 0x7fffffff; /* high word of |x| */

    /* special cases */
    if hx >= 0x4086232b {
        /* if |x| >= 708.39... */
        if x.is_nan() {
            return x;
        }
        if x > 709.782712893383973096 {
            /* overflow if x!=inf */
            x *= f64::from_bits(0x7fe0000000000000);
            return x;
        }
        if x < -708.39641853226410622 {
            /* underflow if x!=-inf */
            force_eval!((f64::from_bits(0xb6a0000000000000) / x) as f32);
            if x < -745.13321910194110842 {
                return 0.0;
            }
        }
    }

    /* argument reduction */
    let k: i32;
    let hi: f64;
    let lo: f64;
    if hx > 0x3fd62e42 {
        /* if |x| > 0.5 ln2 */
        /* if |x| > 0.5 ln2 */
        if hx > 0x3ff0a2b2 {
            /* if |x| > 1.5 ln2 */
            k = (INV_LN2 * x + HALF[sign as usize]) as i32;
        } else {
            k = 1 - sign - sign;
        }
        let kf = k as f64;
        hi = x - kf * LN2_HI; /* k*ln2hi is exact here */
        lo = kf * LN2_LO;
        x = hi - lo;
    } else if hx > 0x3e300000 {
        /* |x| > 2**-14 */
        k = 0;
        hi = x;
        lo = 0.0;
    } else {
        /* raise inexact */
        force_eval!(f64::from_bits(0x7fe0000000000000) + x);
        return 1.0 + x;
    }

    /* x is now in primary range */
    let xx = x * x;
    let c = x - xx * (P1 + xx * (P2 + xx * (P3 + xx * (P4 + xx * P5))));
    let y = 1.0 + (x * c / (2.0 - c) - lo + hi);
    if k == 0 {
        y
    } else {
        scalbn(y, k)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanity_check() {
        assert_eq!(super::exp(1.1), 3.0041660239464334);
    }
}
