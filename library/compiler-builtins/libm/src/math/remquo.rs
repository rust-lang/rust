pub fn remquo(mut x: f64, mut y: f64) -> (f64, isize)
{
    let ux: u64 = x.to_bits();
    let mut uy: u64 = y.to_bits();
    let mut ex = ((ux>>52) & 0x7ff) as isize;
    let mut ey = ((uy>>52) & 0x7ff) as isize;
    let sx = (ux>>63) != 0;
    let sy = (uy>>63) != 0;
    let mut q: u32;
    let mut i: u64;
    let mut uxi: u64 = ux;

    if (uy<<1) == 0 || y.is_nan() || ex == 0x7ff {
        return ((x*y)/(x*y), 0);
    }
    if (ux<<1) == 0 {
        return (x, 0);
    }

    /* normalize x and y */
    if ex == 0 {
        i = uxi << 12;
        while (i>>63) == 0 {
            ex -= 1;
            i <<= 1;
        }
        uxi <<= -ex + 1;
    } else {
        uxi &= (!0) >> 12;
        uxi |= 1 << 52;
    }
    if ey == 0 {
        i = uy<<12;
        while (i>>63) == 0 {
            ey -= 1;
            i <<= 1;
        }
        uy <<= -ey + 1;
    } else {
        uy &= (!0) >> 12;
        uy |= 1 << 52;
    }

    q = 0;

    if ex+1 != ey {
        if ex < ey {
            return (x, 0);
        }
        /* x mod y */
        while ex > ey {
            i = uxi - uy;
            if (i>>63) == 0 {
                uxi = i;
                q += 1;
            }
            uxi <<= 1;
            q <<= 1;
            ex -= 1;
        }
        i = uxi - uy;
        if (i>>63) == 0 {
            uxi = i;
            q += 1;
        }
        if uxi == 0 {
            ex = -60;
        } else {
            while (uxi>>52) == 0 {
                uxi <<= 1;
                ex -= 1;
            }
        }
    }

    /* scale result and decide between |x| and |x|-|y| */
    if ex > 0 {
        uxi -= 1 << 52;
        uxi |= (ex as u64) << 52;
    } else {
        uxi >>= -ex + 1;
    }
    x = f64::from_bits(uxi);
    if sy {
        y = -y;
    }
    if ex == ey || (ex+1 == ey && (2.0*x > y || (2.0*x == y && (q%2) != 0))) {
        x -= y;
        q += 1;
    }
    q &= 0x7fffffff;
    let quo = if sx ^ sy { -(q as isize) } else { q as isize };
    if sx {
        (-x, quo)
    } else {
        (x, quo)
    }
}
