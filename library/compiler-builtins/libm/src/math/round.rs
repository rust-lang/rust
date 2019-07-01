use core::f64;

const TOINT: f64 = 1.0 / f64::EPSILON;

#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn round(mut x: f64) -> f64 {
    let i = x.to_bits();
    let e: u64 = i >> 52 & 0x7ff;
    let mut y: f64;

    if e >= 0x3ff + 52 {
        return x;
    }
    if e < 0x3ff - 1 {
        // raise inexact if x!=0
        force_eval!(x + TOINT);
        return 0.0 * x;
    }
    if i >> 63 != 0 {
        x = -x;
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

#[cfg(test)]
mod tests {
    use super::round;

    #[test]
    fn negative_zero() {
        assert_eq!(round(-0.0_f64).to_bits(), (-0.0_f64).to_bits());
    }
}
