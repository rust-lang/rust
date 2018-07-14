use core::u64;

#[inline]
pub fn fmod(x: f64, y: f64) -> f64 {
    let mut uxi = x.to_bits();
    let mut uyi = y.to_bits();
    let mut ex = (uxi >> 52 & 0x7ff) as i64;
    let mut ey = (uyi >> 52 & 0x7ff) as i64;
    let sx = uxi >> 63;
    let mut i;

    if uyi << 1 == 0 || y.is_nan() || ex == 0x7ff {
        return (x * y) / (x * y);
    }
    if uxi << 1 <= uyi << 1 {
        if uxi << 1 == uyi << 1 {
            return 0.0 * x;
        }
        return x;
    }

    /* normalize x and y */
    if ex == 0 {
        i = uxi << 12;
        while i >> 63 == 0 {
            ex -= 1;
            i <<= 1;
        }
        uxi <<= -ex + 1;
    } else {
        uxi &= u64::MAX >> 12;
        uxi |= 1 << 52;
    }
    if ey == 0 {
        i = uyi << 12;
        while i >> 63 == 0 {
            ey -= 1;
            i <<= 1;
        }
        uyi <<= -ey + 1;
    } else {
        uyi &= u64::MAX >> 12;
        uyi |= 1 << 52;
    }

    /* x mod y */
    while ex > ey {
        i = uxi - uyi;
        if i >> 63 == 0 {
            if i == 0 {
                return 0.0 * x;
            }
            uxi = i;
        }
        uxi <<= 1;
        ex -= 1;
    }
    i = uxi - uyi;
    if i >> 63 == 0 {
        if i == 0 {
            return 0.0 * x;
        }
        uxi = i;
    }
    while uxi >> 52 == 0 {
        uxi <<= 1;
        ex -= 1;
    }

    /* scale result */
    if ex > 0 {
        uxi -= 1 << 52;
        uxi |= (ex as u64) << 52;
    } else {
        uxi >>= -ex + 1;
    }
    uxi |= (sx as u64) << 63;

    f64::from_bits(uxi)
}
