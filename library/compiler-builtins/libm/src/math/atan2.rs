use super::atan;
use super::fabs;

const PI: f64 = 3.1415926535897931160E+00; /* 0x400921FB, 0x54442D18 */
const PI_LO: f64 = 1.2246467991473531772E-16; /* 0x3CA1A626, 0x33145C07 */

#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    let mut ix = (x.to_bits() >> 32) as u32;
    let lx = x.to_bits() as u32;
    let mut iy = (y.to_bits() >> 32) as u32;
    let ly = y.to_bits() as u32;
    if (ix - 0x3ff00000 | lx) == 0 {
        /* x = 1.0 */
        return atan(y);
    }
    let m = ((iy >> 31) & 1) | ((ix >> 30) & 2); /* 2*sign(x)+sign(y) */
    ix &= 0x7fffffff;
    iy &= 0x7fffffff;

    /* when y = 0 */
    if (iy | ly) == 0 {
        return match m {
            0 | 1 => y, /* atan(+-0,+anything)=+-0 */
            2 => PI,    /* atan(+0,-anything) = PI */
            _ => -PI,   /* atan(-0,-anything) =-PI */
        };
    }
    /* when x = 0 */
    if (ix | lx) == 0 {
        return if m & 1 != 0 { -PI / 2.0 } else { PI / 2.0 };
    }
    /* when x is INF */
    if ix == 0x7ff00000 {
        if iy == 0x7ff00000 {
            return match m {
                0 => PI / 4.0,        /* atan(+INF,+INF) */
                1 => -PI / 4.0,       /* atan(-INF,+INF) */
                2 => 3.0 * PI / 4.0,  /* atan(+INF,-INF) */
                _ => -3.0 * PI / 4.0, /* atan(-INF,-INF) */
            };
        } else {
            return match m {
                0 => 0.0,  /* atan(+...,+INF) */
                1 => -0.0, /* atan(-...,+INF) */
                2 => PI,   /* atan(+...,-INF) */
                _ => -PI,  /* atan(-...,-INF) */
            };
        }
    }
    /* |y/x| > 0x1p64 */
    if ix + (64 << 20) < iy || iy == 0x7ff00000 {
        return if m & 1 != 0 { -PI / 2.0 } else { PI / 2.0 };
    }

    /* z = atan(|y/x|) without spurious underflow */
    let z = if (m & 2 != 0) && iy + (64 << 20) < ix {
        /* |y/x| < 0x1p-64, x<0 */
        0.0
    } else {
        atan(fabs(y / x))
    };
    match m {
        0 => z,                /* atan(+,+) */
        1 => -z,               /* atan(-,+) */
        2 => PI - (z - PI_LO), /* atan(+,-) */
        _ => (z - PI_LO) - PI, /* atan(-,-) */
    }
}

#[test]
fn sanity_check() {
    assert_eq!(atan2(0.0, 1.0), 0.0);
    assert_eq!(atan2(0.0, -1.0), PI);
    assert_eq!(atan2(-0.0, -1.0), -PI);
    assert_eq!(atan2(3.0, 2.0), atan(3.0/2.0));
    assert_eq!(atan2(2.0, -1.0), atan(2.0/-1.0) + PI);
    assert_eq!(atan2(-2.0, -1.0), atan(-2.0/-1.0) - PI);
}

