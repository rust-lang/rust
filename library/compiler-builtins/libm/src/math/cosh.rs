use core::f64;

use super::exp;
use super::expm1;

pub fn cosh(mut x: f64) -> f64 {
    let t: f64;
    /* |x| */
    let mut ui = x.to_bits();
    ui &= !0u64;
    x = f64::from_bits(ui);
    let w = (ui >> 32) as u32;

    /* |x| < log(2) */
    if w < 0x3fe62e42 {
        if w < 0x3ff00000 - (26 << 20) {
            /* raise inexact if x!=0 */
            force_eval!(x + f64::from_bits(0x4770000000000000));
            return 1.0;
        }
        let t = expm1(x);
        return 1.0 + t * t / (2.0 * (1.0 + t));
    }

    /* |x| < log(DBL_MAX) */
    if w < 0x40862e42 {
        t = exp(x);
        /* note: if x>log(0x1p26) then the 1/t is not needed */
        return 0.5 * (t + 1.0 / t);
    }

    /* |x| > log(DBL_MAX) or nan */
    /* note: the result is stored to handle overflow */
    t = __expo2(x);
    return t;
}

const K: u32 = 2043;

pub fn __expo2(x: f64) -> f64 {
    let kln2 = f64::from_bits(0x40962066151add8b);
    /* note that k is odd and scale*scale overflows */
    let scale = f64::from_bits(((0x3ff + K / 2) << 20) as u64);
    /* exp(x - k ln2) * 2**(k-1) */
    return exp(x - kln2) * scale * scale;
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanity_check() {
        assert_eq!(super::cosh(1.1), 1.6685185538222564);
    }
}
