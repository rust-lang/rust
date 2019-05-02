use core::f64;

const TOINT: f64 = 1.0 / f64::EPSILON;

#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn round(mut x: f64) -> f64 {
    let (f, i) = (x, x.to_bits());
    let e: u64 = i >> 52 & 0x7ff;
    let mut y: f64;

    if e >= 0x3ff + 52 {
        return x;
    }
    if i >> 63 != 0 {
        x = -x;
    }
    if e < 0x3ff - 1 {
        // raise inexact if x!=0
        force_eval!(x + TOINT);
        return 0.0 * f;
    }
    y = x + TOINT - TOINT - x;
    if y > 0.5 {
        y = y + x - 1.0;
    } else if y <= -0.5 {
        y = y + x + 1.0;
    } else {
        y = y + x;
    }

    if i >> 63 != 0 {
        -y
    } else {
        y
    }
}
