use core::f64;

const TOINT: f64 = 1. / f64::EPSILON;

#[inline]
pub fn ceil(x: f64) -> f64 {
    let u: u64 = x.to_bits();
    let e: i64 = (u >> 52 & 0x7ff) as i64;
    let y: f64;

    if e >= 0x3ff + 52 || x == 0. {
        return x;
    }
    // y = int(x) - x, where int(x) is an integer neighbor of x
    y = if (u >> 63) != 0 {
        x - TOINT + TOINT - x
    } else {
        x + TOINT - TOINT - x
    };
    // special case because of non-nearest rounding modes
    if e <= 0x3ff - 1 {
        force_eval!(y);
        return if (u >> 63) != 0 { -0. } else { 1. };
    }
    if y < 0. {
        x + y + 1.
    } else {
        x + y
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sanity_check() {
        assert_eq!(super::ceil(1.1), 2.0);
        assert_eq!(super::ceil(2.9), 3.0);
    }
}
