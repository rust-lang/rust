use core::{f32, f64};

use super::scalbn;

const ZEROINFNAN: i32 = 0x7ff - 0x3ff - 52 - 1;

struct Num {
    m: u64,
    e: i32,
    sign: i32,
}

fn normalize(x: f64) -> Num {
    let x1p63: f64 = f64::from_bits(0x43e0000000000000); // 0x1p63 === 2 ^ 63

    let mut ix: u64 = x.to_bits();
    let mut e: i32 = (ix >> 52) as i32;
    let sign: i32 = e & 0x800;
    e &= 0x7ff;
    if e == 0 {
        ix = (x * x1p63).to_bits();
        e = (ix >> 52) as i32 & 0x7ff;
        e = if e != 0 { e - 63 } else { 0x800 };
    }
    ix &= (1 << 52) - 1;
    ix |= 1 << 52;
    ix <<= 1;
    e -= 0x3ff + 52 + 1;
    Num { m: ix, e, sign }
}

fn mul(x: u64, y: u64) -> (u64, u64) {
    let t1: u64;
    let t2: u64;
    let t3: u64;
    let xlo: u64 = x as u32 as u64;
    let xhi: u64 = x >> 32;
    let ylo: u64 = y as u32 as u64;
    let yhi: u64 = y >> 32;

    t1 = xlo * ylo;
    t2 = xlo * yhi + xhi * ylo;
    t3 = xhi * yhi;
    let lo = t1.wrapping_add(t2 << 32);
    let hi = t3 + (t2 >> 32) + (t1 > lo) as u64;
    (hi, lo)
}

/// Floating multiply add (f64)
///
/// Computes `(x*y)+z`, rounded as one ternary operation:
/// Computes the value (as if) to infinite precision and rounds once to the result format,
/// according to the rounding mode characterized by the value of FLT_ROUNDS.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    let x1p63: f64 = f64::from_bits(0x43e0000000000000); // 0x1p63 === 2 ^ 63
    let x0_ffffff8p_63 = f64::from_bits(0x3bfffffff0000000); // 0x0.ffffff8p-63

    /* normalize so top 10bits and last bit are 0 */
    let nx = normalize(x);
    let ny = normalize(y);
    let nz = normalize(z);

    if nx.e >= ZEROINFNAN || ny.e >= ZEROINFNAN {
        return x * y + z;
    }
    if nz.e >= ZEROINFNAN {
        if nz.e > ZEROINFNAN {
            /* z==0 */
            return x * y + z;
        }
        return z;
    }

    /* mul: r = x*y */
    let zhi: u64;
    let zlo: u64;
    let (mut rhi, mut rlo) = mul(nx.m, ny.m);
    /* either top 20 or 21 bits of rhi and last 2 bits of rlo are 0 */

    /* align exponents */
    let mut e: i32 = nx.e + ny.e;
    let mut d: i32 = nz.e - e;
    /* shift bits z<<=kz, r>>=kr, so kz+kr == d, set e = e+kr (== ez-kz) */
    if d > 0 {
        if d < 64 {
            zlo = nz.m << d;
            zhi = nz.m >> (64 - d);
        } else {
            zlo = 0;
            zhi = nz.m;
            e = nz.e - 64;
            d -= 64;
            if d == 0 {
            } else if d < 64 {
                rlo = rhi << (64 - d) | rlo >> d | ((rlo << (64 - d)) != 0) as u64;
                rhi = rhi >> d;
            } else {
                rlo = 1;
                rhi = 0;
            }
        }
    } else {
        zhi = 0;
        d = -d;
        if d == 0 {
            zlo = nz.m;
        } else if d < 64 {
            zlo = nz.m >> d | ((nz.m << (64 - d)) != 0) as u64;
        } else {
            zlo = 1;
        }
    }

    /* add */
    let mut sign: i32 = nx.sign ^ ny.sign;
    let samesign: bool = (sign ^ nz.sign) == 0;
    let mut nonzero: i32 = 1;
    if samesign {
        /* r += z */
        rlo = rlo.wrapping_add(zlo);
        rhi += zhi + (rlo < zlo) as u64;
    } else {
        /* r -= z */
        let (res, borrow) = rlo.overflowing_sub(zlo);
        rlo = res;
        rhi = rhi.wrapping_sub(zhi.wrapping_add(borrow as u64));
        if (rhi >> 63) != 0 {
            rlo = (-(rlo as i64)) as u64;
            rhi = (-(rhi as i64)) as u64 - (rlo != 0) as u64;
            sign = (sign == 0) as i32;
        }
        nonzero = (rhi != 0) as i32;
    }

    /* set rhi to top 63bit of the result (last bit is sticky) */
    if nonzero != 0 {
        e += 64;
        d = rhi.leading_zeros() as i32 - 1;
        /* note: d > 0 */
        rhi = rhi << d | rlo >> (64 - d) | ((rlo << d) != 0) as u64;
    } else if rlo != 0 {
        d = rlo.leading_zeros() as i32 - 1;
        if d < 0 {
            rhi = rlo >> 1 | (rlo & 1);
        } else {
            rhi = rlo << d;
        }
    } else {
        /* exact +-0 */
        return x * y + z;
    }
    e -= d;

    /* convert to double */
    let mut i: i64 = rhi as i64; /* i is in [1<<62,(1<<63)-1] */
    if sign != 0 {
        i = -i;
    }
    let mut r: f64 = i as f64; /* |r| is in [0x1p62,0x1p63] */

    if e < -1022 - 62 {
        /* result is subnormal before rounding */
        if e == -1022 - 63 {
            let mut c: f64 = x1p63;
            if sign != 0 {
                c = -c;
            }
            if r == c {
                /* min normal after rounding, underflow depends
                on arch behaviour which can be imitated by
                a double to float conversion */
                let fltmin: f32 = (x0_ffffff8p_63 * f32::MIN_POSITIVE as f64 * r) as f32;
                return f64::MIN_POSITIVE / f32::MIN_POSITIVE as f64 * fltmin as f64;
            }
            /* one bit is lost when scaled, add another top bit to
            only round once at conversion if it is inexact */
            if (rhi << 53) != 0 {
                i = (rhi >> 1 | (rhi & 1) | 1 << 62) as i64;
                if sign != 0 {
                    i = -i;
                }
                r = i as f64;
                r = 2. * r - c; /* remove top bit */

                /* raise underflow portably, such that it
                cannot be optimized away */
                {
                    let tiny: f64 = f64::MIN_POSITIVE / f32::MIN_POSITIVE as f64 * r;
                    r += (tiny * tiny) * (r - r);
                }
            }
        } else {
            /* only round once when scaled */
            d = 10;
            i = ((rhi >> d | ((rhi << (64 - d)) != 0) as u64) << d) as i64;
            if sign != 0 {
                i = -i;
            }
            r = i as f64;
        }
    }
    scalbn(r, e)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fma_segfault() {
        // These two inputs cause fma to segfault on release due to overflow:
        assert_eq!(
            fma(
                -0.0000000000000002220446049250313,
                -0.0000000000000002220446049250313,
                -0.0000000000000002220446049250313
            ),
            -0.00000000000000022204460492503126,
        );

        let result = fma(-0.992, -0.992, -0.992);
        //force rounding to storage format on x87 to prevent superious errors.
        #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
        let result = force_eval!(result);
        assert_eq!(result, -0.007936000000000007,);
    }

    #[test]
    fn fma_sbb() {
        assert_eq!(
            fma(-(1.0 - f64::EPSILON), f64::MIN, f64::MIN),
            -3991680619069439e277
        );
    }
}
