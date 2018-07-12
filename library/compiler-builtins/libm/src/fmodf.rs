use core::u32;

use isnanf;

pub fn fmodf(x: f32, y: f32) -> f32 {
    let mut uxi = x.to_bits();
    let mut uyi = y.to_bits();
    let mut ex = (uxi >> 23 & 0xff) as i32;
    let mut ey = (uyi >> 23 & 0xff) as i32;
    let sx = uxi & 0x80000000;
    let mut i;

    if uyi << 1 == 0 || isnanf(y) || ex == 0xff {
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
        i = uxi << 9;
        while i >> 31 == 0 {
            ex -= 1;
            i <<= 1;
        }

        uxi <<= -ex + 1;
    } else {
        uxi &= u32::MAX >> 9;
        uxi |= 1 << 23;
    }

    if ey == 0 {
        i = uyi << 9;
        while i >> 31 == 0 {
            ey -= 1;
            i <<= 1;
        }

        uyi <<= -ey + 1;
    } else {
        uyi &= u32::MAX >> 9;
        uyi |= 1 << 23;
    }

    /* x mod y */
    while ex > ey {
        i = uxi - uyi;
        if i >> 31 == 0 {
            if i == 0 {
                return 0.0 * x;
            }
            uxi = i;
        }
        uxi <<= 1;

        ex -= 1;
    }

    i = uxi - uyi;
    if i >> 31 == 0 {
        if i == 0 {
            return 0.0 * x;
        }
        uxi = i;
    }

    while uxi >> 23 == 0 {
        uxi <<= 1;
        ex -= 1;
    }

    /* scale result up */
    if ex > 0 {
        uxi -= 1 << 23;
        uxi |= (ex as u32) << 23;
    } else {
        uxi >>= -ex + 1;
    }
    uxi |= sx;

    f32::from_bits(uxi)
}
