/* SPDX-License-Identifier: MIT */
/* origin: musl src/math/fma.c. Ported to generic Rust algorithm in 2025, TG. */

use super::support::{DInt, FpResult, HInt, IntTy, Round, Status};
use super::{CastFrom, CastInto, Float, Int, MinInt};

/// Fused multiply add (f64)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    fma_round(x, y, z, Round::Nearest).val
}

/// Fused multiply add (f128)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaf128(x: f128, y: f128, z: f128) -> f128 {
    fma_round(x, y, z, Round::Nearest).val
}

/// Fused multiply-add that works when there is not a larger float size available. Computes
/// `(x * y) + z`.
pub fn fma_round<F>(x: F, y: F, z: F, _round: Round) -> FpResult<F>
where
    F: Float,
    F: CastFrom<F::SignedInt>,
    F: CastFrom<i8>,
    F::Int: HInt,
    u32: CastInto<F::Int>,
{
    let one = IntTy::<F>::ONE;
    let zero = IntTy::<F>::ZERO;

    // Normalize such that the top of the mantissa is zero and we have a guard bit.
    let nx = Norm::from_float(x);
    let ny = Norm::from_float(y);
    let nz = Norm::from_float(z);

    if nx.is_zero_nan_inf() || ny.is_zero_nan_inf() {
        // Value will overflow, defer to non-fused operations.
        return FpResult::ok(x * y + z);
    }

    if nz.is_zero_nan_inf() {
        if nz.is_zero() {
            // Empty add component means we only need to multiply.
            return FpResult::ok(x * y);
        }
        // `z` is NaN or infinity, which sets the result.
        return FpResult::ok(z);
    }

    // multiply: r = x * y
    let zhi: F::Int;
    let zlo: F::Int;
    let (mut rlo, mut rhi) = nx.m.widen_mul(ny.m).lo_hi();

    // Exponent result of multiplication
    let mut e: i32 = nx.e + ny.e;
    // Needed shift to align `z` to the multiplication result
    let mut d: i32 = nz.e - e;
    let sbits = F::BITS as i32;

    // Scale `z`. Shift `z <<= kz`, `r >>= kr`, so `kz+kr == d`, set `e = e+kr` (== ez-kz)
    if d > 0 {
        // The magnitude of `z` is larger than `x * y`
        if d < sbits {
            // Maximum shift of one `F::BITS` means shifted `z` will fit into `2 * F::BITS`. Shift
            // it into `(zhi, zlo)`. No exponent adjustment necessary.
            zlo = nz.m << d;
            zhi = nz.m >> (sbits - d);
        } else {
            // Shift larger than `sbits`, `z` only needs the top half `zhi`. Place it there (acts
            // as a shift by `sbits`).
            zlo = zero;
            zhi = nz.m;
            d -= sbits;

            // `z`'s exponent is large enough that it now needs to be taken into account.
            e = nz.e - sbits;

            if d == 0 {
                // Exactly `sbits`, nothing to do
            } else if d < sbits {
                // Remaining shift fits within `sbits`. Leave `z` in place, shift `x * y`
                rlo = (rhi << (sbits - d)) | (rlo >> d);
                // Set the sticky bit
                rlo |= IntTy::<F>::from((rlo << (sbits - d)) != zero);
                rhi = rhi >> d;
            } else {
                // `z`'s magnitude is enough that `x * y` is irrelevant. It was nonzero, so set
                // the sticky bit.
                rlo = one;
                rhi = zero;
            }
        }
    } else {
        // `z`'s magnitude once shifted fits entirely within `zlo`
        zhi = zero;
        d = -d;
        if d == 0 {
            // No shift needed
            zlo = nz.m;
        } else if d < sbits {
            // Shift s.t. `nz.m` fits into `zlo`
            let sticky = IntTy::<F>::from((nz.m << (sbits - d)) != zero);
            zlo = (nz.m >> d) | sticky;
        } else {
            // Would be entirely shifted out, only set the sticky bit
            zlo = one;
        }
    }

    /* addition */

    let mut neg = nx.neg ^ ny.neg;
    let samesign: bool = !neg ^ nz.neg;
    let mut rhi_nonzero = true;

    if samesign {
        // r += z
        rlo = rlo.wrapping_add(zlo);
        rhi += zhi + IntTy::<F>::from(rlo < zlo);
    } else {
        // r -= z
        let (res, borrow) = rlo.overflowing_sub(zlo);
        rlo = res;
        rhi = rhi.wrapping_sub(zhi.wrapping_add(IntTy::<F>::from(borrow)));
        if (rhi >> (F::BITS - 1)) != zero {
            rlo = rlo.signed().wrapping_neg().unsigned();
            rhi = rhi.signed().wrapping_neg().unsigned() - IntTy::<F>::from(rlo != zero);
            neg = !neg;
        }
        rhi_nonzero = rhi != zero;
    }

    /* Construct result */

    // Shift result into `rhi`, left-aligned. Last bit is sticky
    if rhi_nonzero {
        // `d` > 0, need to shift both `rhi` and `rlo` into result
        e += sbits;
        d = rhi.leading_zeros() as i32 - 1;
        rhi = (rhi << d) | (rlo >> (sbits - d));
        // Update sticky
        rhi |= IntTy::<F>::from((rlo << d) != zero);
    } else if rlo != zero {
        // `rhi` is zero, `rlo` is the entire result and needs to be shifted
        d = rlo.leading_zeros() as i32 - 1;
        if d < 0 {
            // Shift and set sticky
            rhi = (rlo >> 1) | (rlo & one);
        } else {
            rhi = rlo << d;
        }
    } else {
        // exact +/- 0.0
        return FpResult::ok(x * y + z);
    }

    e -= d;

    // Use int->float conversion to populate the significand.
    // i is in [1 << (BITS - 2), (1 << (BITS - 1)) - 1]
    let mut i: F::SignedInt = rhi.signed();

    if neg {
        i = -i;
    }

    // `|r|` is in `[0x1p62,0x1p63]` for `f64`
    let mut r: F = F::cast_from_lossy(i);

    /* Account for subnormal and rounding */

    // Unbiased exponent for the maximum value of `r`
    let max_pow = F::BITS - 1 + F::EXP_BIAS;

    let mut status = Status::OK;

    if e < -(max_pow as i32 - 2) {
        // Result is subnormal before rounding
        if e == -(max_pow as i32 - 1) {
            let mut c = F::from_parts(false, max_pow, zero);
            if neg {
                c = -c;
            }

            if r == c {
                // Min normal after rounding,
                status.set_underflow(true);
                r = F::MIN_POSITIVE_NORMAL.copysign(r);
                return FpResult::new(r, status);
            }

            if (rhi << (F::SIG_BITS + 1)) != zero {
                // Account for truncated bits. One bit will be lost in the `scalbn` call, add
                // another top bit to avoid double rounding if inexact.
                let iu: F::Int = (rhi >> 1) | (rhi & one) | (one << (F::BITS - 2));
                i = iu.signed();

                if neg {
                    i = -i;
                }

                r = F::cast_from_lossy(i);

                // Remove the top bit
                r = F::cast_from(2i8) * r - c;
                status.set_underflow(true);
            }
        } else {
            // Only round once when scaled
            d = F::EXP_BITS as i32 - 1;
            let sticky = IntTy::<F>::from(rhi << (F::BITS as i32 - d) != zero);
            i = (((rhi >> d) | sticky) << d).signed();

            if neg {
                i = -i;
            }

            r = F::cast_from_lossy(i);
        }
    }

    // Use our exponent to scale the final value.
    FpResult::new(super::generic::scalbn(r, e), status)
}

/// Representation of `F` that has handled subnormals.
#[derive(Clone, Copy, Debug)]
struct Norm<F: Float> {
    /// Normalized significand with one guard bit, unsigned.
    m: F::Int,
    /// Exponent of the mantissa such that `m * 2^e = x`. Accounts for the shift in the mantissa
    /// and the guard bit; that is, 1.0 will normalize as `m = 1 << 53` and `e = -53`.
    e: i32,
    neg: bool,
}

impl<F: Float> Norm<F> {
    /// Unbias the exponent and account for the mantissa's precision, including the guard bit.
    const EXP_UNBIAS: u32 = F::EXP_BIAS + F::SIG_BITS + 1;

    /// Values greater than this had a saturated exponent (infinity or NaN), OR were zero and we
    /// adjusted the exponent such that it exceeds this threashold.
    const ZERO_INF_NAN: u32 = F::EXP_SAT - Self::EXP_UNBIAS;

    fn from_float(x: F) -> Self {
        let mut ix = x.to_bits();
        let mut e = x.ex() as i32;
        let neg = x.is_sign_negative();
        if e == 0 {
            // Normalize subnormals by multiplication
            let scale_i = F::BITS - 1;
            let scale_f = F::from_parts(false, scale_i + F::EXP_BIAS, F::Int::ZERO);
            let scaled = x * scale_f;
            ix = scaled.to_bits();
            e = scaled.ex() as i32;
            e = if e == 0 {
                // If the exponent is still zero, the input was zero. Artifically set this value
                // such that the final `e` will exceed `ZERO_INF_NAN`.
                1 << F::EXP_BITS
            } else {
                // Otherwise, account for the scaling we just did.
                e - scale_i as i32
            };
        }

        e -= Self::EXP_UNBIAS as i32;

        // Absolute  value, set the implicit bit, and shift to create a guard bit
        ix &= F::SIG_MASK;
        ix |= F::IMPLICIT_BIT;
        ix <<= 1;

        Self { m: ix, e, neg }
    }

    /// True if the value was zero, infinity, or NaN.
    fn is_zero_nan_inf(self) -> bool {
        self.e >= Self::ZERO_INF_NAN as i32
    }

    /// The only value we have
    fn is_zero(self) -> bool {
        // The only exponent that strictly exceeds this value is our sentinel value for zero.
        self.e > Self::ZERO_INF_NAN as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the generic `fma_round` algorithm for a given float.
    fn spec_test<F>()
    where
        F: Float,
        F: CastFrom<F::SignedInt>,
        F: CastFrom<i8>,
        F::Int: HInt,
        u32: CastInto<F::Int>,
    {
        let x = F::from_bits(F::Int::ONE);
        let y = F::from_bits(F::Int::ONE);
        let z = F::ZERO;

        let fma = |x, y, z| fma_round(x, y, z, Round::Nearest).val;

        // 754-2020 says "When the exact result of (a × b) + c is non-zero yet the result of
        // fusedMultiplyAdd is zero because of rounding, the zero result takes the sign of the
        // exact result"
        assert_biteq!(fma(x, y, z), F::ZERO);
        assert_biteq!(fma(x, -y, z), F::NEG_ZERO);
        assert_biteq!(fma(-x, y, z), F::NEG_ZERO);
        assert_biteq!(fma(-x, -y, z), F::ZERO);
    }

    #[test]
    fn spec_test_f32() {
        spec_test::<f32>();
    }

    #[test]
    fn spec_test_f64() {
        spec_test::<f64>();

        let expect_underflow = [
            (
                hf64!("0x1.0p-1070"),
                hf64!("0x1.0p-1070"),
                hf64!("0x1.ffffffffffffp-1023"),
                hf64!("0x0.ffffffffffff8p-1022"),
            ),
            (
                // FIXME: we raise underflow but this should only be inexact (based on C and
                // `rustc_apfloat`).
                hf64!("0x1.0p-1070"),
                hf64!("0x1.0p-1070"),
                hf64!("-0x1.0p-1022"),
                hf64!("-0x1.0p-1022"),
            ),
        ];

        for (x, y, z, res) in expect_underflow {
            let FpResult { val, status } = fma_round(x, y, z, Round::Nearest);
            assert_biteq!(val, res);
            assert_eq!(status, Status::UNDERFLOW);
        }
    }

    #[test]
    #[cfg(f128_enabled)]
    fn spec_test_f128() {
        spec_test::<f128>();
    }

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
        assert_eq!(fma(-(1.0 - f64::EPSILON), f64::MIN, f64::MIN), -3991680619069439e277);
    }

    #[test]
    fn fma_underflow() {
        assert_eq!(fma(1.1102230246251565e-16, -9.812526705433188e-305, 1.0894e-320), 0.0,);
    }
}
