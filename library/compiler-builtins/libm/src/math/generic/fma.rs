use core::{f32, f64};

use super::super::support::{DInt, HInt, IntTy};
use super::super::{CastFrom, CastInto, Float, Int, MinInt};

const ZEROINFNAN: i32 = 0x7ff - 0x3ff - 52 - 1;

/// Fused multiply-add that works when there is not a larger float size available. Currently this
/// is still specialized only for `f64`. Computes `(x * y) + z`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fma<F>(x: F, y: F, z: F) -> F
where
    F: Float + FmaHelper,
    F: CastFrom<F::SignedInt>,
    F: CastFrom<i8>,
    F::Int: HInt,
    u32: CastInto<F::Int>,
{
    let one = IntTy::<F>::ONE;
    let zero = IntTy::<F>::ZERO;
    let magic = F::from_parts(false, F::BITS - 1 + F::EXP_BIAS, zero);

    /* normalize so top 10bits and last bit are 0 */
    let nx = Norm::from_float(x);
    let ny = Norm::from_float(y);
    let nz = Norm::from_float(z);

    if nx.e >= ZEROINFNAN || ny.e >= ZEROINFNAN {
        return x * y + z;
    }
    if nz.e >= ZEROINFNAN {
        if nz.e > ZEROINFNAN {
            /* z==0 */
            return x * y;
        }
        return z;
    }

    /* mul: r = x*y */
    let zhi: F::Int;
    let zlo: F::Int;
    let (mut rlo, mut rhi) = nx.m.widen_mul(ny.m).lo_hi();

    /* either top 20 or 21 bits of rhi and last 2 bits of rlo are 0 */

    /* align exponents */
    let mut e: i32 = nx.e + ny.e;
    let mut d: i32 = nz.e - e;
    let sbits = F::BITS as i32;

    /* shift bits z<<=kz, r>>=kr, so kz+kr == d, set e = e+kr (== ez-kz) */
    if d > 0 {
        if d < sbits {
            zlo = nz.m << d;
            zhi = nz.m >> (sbits - d);
        } else {
            zlo = zero;
            zhi = nz.m;
            e = nz.e - sbits;
            d -= sbits;
            if d == 0 {
            } else if d < sbits {
                rlo = (rhi << (sbits - d))
                    | (rlo >> d)
                    | IntTy::<F>::from((rlo << (sbits - d)) != zero);
                rhi = rhi >> d;
            } else {
                rlo = one;
                rhi = zero;
            }
        }
    } else {
        zhi = zero;
        d = -d;
        if d == 0 {
            zlo = nz.m;
        } else if d < sbits {
            zlo = (nz.m >> d) | IntTy::<F>::from((nz.m << (sbits - d)) != zero);
        } else {
            zlo = one;
        }
    }

    /* add */
    let mut neg = nx.neg ^ ny.neg;
    let samesign: bool = !neg ^ nz.neg;
    let mut nonzero: i32 = 1;
    if samesign {
        /* r += z */
        rlo = rlo.wrapping_add(zlo);
        rhi += zhi + IntTy::<F>::from(rlo < zlo);
    } else {
        /* r -= z */
        let (res, borrow) = rlo.overflowing_sub(zlo);
        rlo = res;
        rhi = rhi.wrapping_sub(zhi.wrapping_add(IntTy::<F>::from(borrow)));
        if (rhi >> (F::BITS - 1)) != zero {
            rlo = rlo.signed().wrapping_neg().unsigned();
            rhi = rhi.signed().wrapping_neg().unsigned() - IntTy::<F>::from(rlo != zero);
            neg = !neg;
        }
        nonzero = (rhi != zero) as i32;
    }

    /* set rhi to top 63bit of the result (last bit is sticky) */
    if nonzero != 0 {
        e += sbits;
        d = rhi.leading_zeros() as i32 - 1;
        /* note: d > 0 */
        rhi = (rhi << d) | (rlo >> (sbits - d)) | IntTy::<F>::from((rlo << d) != zero);
    } else if rlo != zero {
        d = rlo.leading_zeros() as i32 - 1;
        if d < 0 {
            rhi = (rlo >> 1) | (rlo & one);
        } else {
            rhi = rlo << d;
        }
    } else {
        /* exact +-0 */
        return x * y + z;
    }
    e -= d;

    /* convert to double */
    let mut i: F::SignedInt = rhi.signed(); /* i is in [1<<62,(1<<63)-1] */
    if neg {
        i = -i;
    }

    let mut r: F = F::cast_from_lossy(i); /* |r| is in [0x1p62,0x1p63] */

    if e < -(F::EXP_BIAS as i32 - 1) - (sbits - 2) {
        /* result is subnormal before rounding */
        if e == -(F::EXP_BIAS as i32 - 1) - (sbits - 1) {
            let mut c: F = magic;
            if neg {
                c = -c;
            }
            if r == c {
                /* min normal after rounding, underflow depends
                 * on arch behaviour which can be imitated by
                 * a double to float conversion */
                return r.raise_underflow();
            }
            /* one bit is lost when scaled, add another top bit to
             * only round once at conversion if it is inexact */
            if (rhi << F::SIG_BITS) != zero {
                let iu: F::Int = (rhi >> 1) | (rhi & one) | (one << 62);
                i = iu.signed();
                if neg {
                    i = -i;
                }
                r = F::cast_from_lossy(i);
                r = F::cast_from(2i8) * r - c; /* remove top bit */

                /* raise underflow portably, such that it
                 * cannot be optimized away */
                r += r.raise_underflow2();
            }
        } else {
            /* only round once when scaled */
            d = 10;
            i = (((rhi >> d) | IntTy::<F>::from(rhi << (F::BITS as i32 - d) != zero)) << d)
                .signed();
            if neg {
                i = -i;
            }
            r = F::cast_from(i);
        }
    }

    super::scalbn(r, e)
}

/// Representation of `F` that has handled subnormals.
struct Norm<F: Float> {
    /// Normalized significand with one guard bit.
    m: F::Int,
    /// Unbiased exponent, normalized.
    e: i32,
    neg: bool,
}

impl<F: Float> Norm<F> {
    fn from_float(x: F) -> Self {
        let mut ix = x.to_bits();
        let mut e = x.exp() as i32;
        let neg = x.is_sign_negative();
        if e == 0 {
            // Normalize subnormals by multiplication
            let magic = F::from_parts(false, F::BITS - 1 + F::EXP_BIAS, F::Int::ZERO);
            let scaled = x * magic;
            ix = scaled.to_bits();
            e = scaled.exp() as i32;
            e = if e != 0 { e - (F::BITS as i32 - 1) } else { 0x800 };
        }

        e -= F::EXP_BIAS as i32 + 52 + 1;

        ix &= F::SIG_MASK;
        ix |= F::IMPLICIT_BIT;
        ix <<= 1; // add a guard bit

        Self { m: ix, e, neg }
    }
}

/// Type-specific helpers that are not needed outside of fma.
pub trait FmaHelper {
    fn raise_underflow(self) -> Self;
    fn raise_underflow2(self) -> Self;
}

impl FmaHelper for f64 {
    fn raise_underflow(self) -> Self {
        let x0_ffffff8p_63 = f64::from_bits(0x3bfffffff0000000); // 0x0.ffffff8p-63
        let fltmin: f32 = (x0_ffffff8p_63 * f32::MIN_POSITIVE as f64 * self) as f32;
        f64::MIN_POSITIVE / f32::MIN_POSITIVE as f64 * fltmin as f64
    }

    fn raise_underflow2(self) -> Self {
        /* raise underflow portably, such that it
         * cannot be optimized away */
        let tiny: f64 = f64::MIN_POSITIVE / f32::MIN_POSITIVE as f64 * self;
        (tiny * tiny) * (self - self)
    }
}
