/* SPDX-License-Identifier: MIT OR Apache-2.0
 * origin: original implementation, 2026 (TG) */

use crate::support::{CastFrom, Float, Int, unbounded_shr_u64};

/// We use a a U21.43 fixed-point representation when needed.
const FIXED_FRAC_BITS: u32 = 43;

/// Floating multiply add (f16)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn fmaf16(x: f16, y: f16, z: f16) -> f16 {
    let ix = x.to_bits() & !f16::SIGN_MASK;
    let iy = y.to_bits() & !f16::SIGN_MASK;
    let iz = z.to_bits() & !f16::SIGN_MASK;

    let xneg = x.is_sign_negative();
    let yneg = y.is_sign_negative();
    let zneg = z.is_sign_negative();

    let mneg = xneg ^ yneg;

    if ix == 0 || ix >= f16::EXP_MASK || iy == 0 || iy >= f16::EXP_MASK {
        // Value will overflow, defer to non-fused operations.
        return x * y + z;
    }

    if iz == 0 {
        // Empty add component means we only need to multiply.
        return x * y;
    }

    if iz >= f16::EXP_MASK {
        // `z` is NaN or infinity, which sets the result.
        return z;
    }

    let mut xexp = x.ex();
    let mut yexp = y.ex();
    let mut zexp = z.ex();

    let mut xsig = ix & f16::SIG_MASK;
    let mut ysig = iy & f16::SIG_MASK;
    let mut zsig = iz & f16::SIG_MASK;

    // If not subnormal, set the implicit bit
    if xexp != 0 {
        xsig |= f16::IMPLICIT_BIT;
    }
    if yexp != 0 {
        ysig |= f16::IMPLICIT_BIT;
    }
    if zexp != 0 {
        zsig |= f16::IMPLICIT_BIT;
    }

    // A biased exponent of 1 (min normal) and 0 (subnormal) have the same real exponent, so
    // adjust for this. Bias is now 14 rather than 15.
    xexp = xexp.saturating_sub(1);
    yexp = yexp.saturating_sub(1);
    zexp = zexp.saturating_sub(1);
    let adjbias = f16::EXP_BIAS - 1;

    // Exponent after multiplication. Bias doubles to 28.
    let mexp = xexp + yexp;
    let mbias = adjbias * 2;

    // Exit now if we know the result will overflow. We need to keep one beyond the infinite
    // exponent in case the addition rounds down to a finite number.
    //
    // Note that `EXP_MAX` (i.e. max finite) represents infinity here because our values are
    // acting with a bias of 14.
    let inf_exp = mbias + f16::EXP_MAX.unsigned();
    if mexp > inf_exp + 1 {
        if mneg {
            return f16::NEG_INFINITY;
        } else {
            return f16::INFINITY;
        }
    }

    // Multiplication moves the explicit 1 from the 11th bit to the 22nd bit.
    let m = u32::from(xsig) * u32::from(ysig);
    let mut m64 = u64::from(m);

    // The entire dynamic range of an `f16` fits into a `u64`. Shift based on the exponent to
    // create a U21.43 fixed-point value. At the maximum exponent, there are five zeros before
    // the explicit leading 1 (intentional so this truncates to the final repr).
    if let Some(mshift) = mexp.checked_sub(5) {
        debug_assert_eq!(
            unbounded_shr_u64(m64, 64 - mshift),
            0,
            "data shifted out {m} {mshift}"
        );
        m64 <<= mshift;
    } else {
        // The lower few bits here would be on the order of 2^-43, which is too small to show up
        // in a result significand. Just squash them to a sticky bit.
        let sticky = m64 & 0b11111 != 0;
        m64 >>= 5 - mexp;
        m64 |= u64::from(sticky);
    }

    // Shift z to U21.43 as well.
    let zshift = zexp + FIXED_FRAC_BITS - f16::SIG_BITS - adjbias;
    let z64 = u64::from(zsig) << zshift;

    let sub = mneg ^ zneg;

    let rneg;
    let r64 = if sub {
        if m64 > z64 {
            rneg = mneg;
            m64.wrapping_sub(z64)
        } else if m64 == z64 {
            rneg = false;
            m64.wrapping_sub(z64)
        } else {
            rneg = zneg;
            z64.wrapping_sub(m64)
        }
    } else {
        rneg = mneg;
        m64 + z64
    };

    let sign = if rneg { -1.0 } else { 1.0 };
    f16_from_u21_43(r64).copysign(sign)
}

/// Turn a U21.43 value into an f16 with positive sign.
fn f16_from_u21_43(mut r64: u64) -> f16 {
    let extra_bits = 64 - 16;
    let max_finite_lz = 64 - f16::SIG_BITS - extra_bits - 1; // 5

    // Check for overflow to infinity after addition, return before checking lz.
    if r64 & (u64::MAX << (64 - max_finite_lz)) != 0 {
        return f16::INFINITY;
    }

    // Shift the fixed point to floating point. There are 5 leading zeros before the largest
    // finite value's explicit one.
    //
    // We want `rexp` as one less than the actual value to be stored because it gets added to
    // a value with the leading one set. This value and the shift are clamped so subnormals
    // don't become normalized.
    let exp_max_biased_m1 = f16::EXP_MAX.unsigned() + f16::EXP_BIAS - 1; // 29
    let lz = r64.leading_zeros();
    let rexp = (exp_max_biased_m1 + max_finite_lz).saturating_sub(lz);
    let shift = exp_max_biased_m1 - rexp;
    r64 <<= shift;

    // Round up if the round bit (one past significand end) is set and any trailing bit is set,
    // or if the preceding bit is set.
    let round_bit = 1u64 << (extra_bits - 1);
    let up_mask = ((1u64 << (extra_bits + 1)) - 1) & !round_bit;
    let round_up = r64 & round_bit != 0 && r64 & up_mask != 0;
    let round_up = u16::from(round_up);

    // Truncate then round. Automatically accounts for subnormals with the unset explicit decimal
    // bit, since `rexp` is one less than the actual biased value.
    let mut r = (r64 >> extra_bits) as u16;
    r += u16::cast_from(rexp) << f16::SIG_BITS;
    r += round_up;

    f16::from_bits(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_fixed() {
        // Move 1.xx... floating point to 1.xx... fixed point
        let shift_to_one = |x: u16| u64::from(x) << (FIXED_FRAC_BITS - f16::SIG_BITS);
        let top_sig = f16::IMPLICIT_BIT;
        let max_sig = f16::IMPLICIT_BIT | f16::SIG_MASK;

        // Basic values
        let one = shift_to_one(top_sig);
        let max = shift_to_one(max_sig) << 15;
        let inf = shift_to_one(max_sig + 1) << 15;
        let min_norm = one >> 14;
        let max_sub = shift_to_one(f16::SIG_MASK) >> 14;
        let min_sub = shift_to_one(1) >> 14;

        assert_biteq!(f16_from_u21_43(0), 0.0f16);
        assert_biteq!(f16_from_u21_43(one), 1.0f16);
        assert_biteq!(f16_from_u21_43(max), f16::MAX);
        assert_biteq!(f16_from_u21_43(inf), f16::INFINITY);
        assert_biteq!(f16_from_u21_43(min_norm), f16::MIN_POSITIVE_NORMAL);
        assert_biteq!(f16_from_u21_43(max_sub), f16::from_bits(f16::SIG_MASK));
        assert_biteq!(f16_from_u21_43(min_sub), f16::MIN_POSITIVE_SUBNORMAL);

        // Masks centered around 1 to add a rounding
        let mask_r = shift_to_one(0b1) >> 1; // round bit
        let mask_rg = shift_to_one(0b11) >> 2; // round + guard
        let mask_rgs = shift_to_one(0b111) >> 3; // round + guard + sticky
        let mask_rs = shift_to_one(0b101) >> 3; // round + sticky
        let mask_rs2 = shift_to_one(0b1000_0001) >> 8; // round + part of sticky

        let signed_shift = |val: u64, shift: i32| {
            if shift >= 0 {
                val << shift
            } else {
                val >> -shift
            }
        };

        let check_round = |fixed: u64, shift: i32, lsb_set: bool, down: f16, up: f16| {
            // Masks that will cause rounding down
            let mdown = if lsb_set { &[0][..] } else { &[0, mask_r][..] };
            // Masks that will cause rounding up
            let mup = if lsb_set {
                &[mask_r, mask_rg, mask_rgs, mask_rs, mask_rs2][..]
            } else {
                &[mask_rg, mask_rgs, mask_rs, mask_rs2][..]
            };

            for (i, mask) in mdown.iter().enumerate() {
                let bits = fixed | signed_shift(*mask, shift);
                assert_biteq!(f16_from_u21_43(bits), down, "{bits:#066b} {i}");
            }

            for (i, mask) in mup.iter().enumerate() {
                let bits = fixed | signed_shift(*mask, shift);
                assert_biteq!(f16_from_u21_43(bits), up, "{bits:#066b} {i}");
            }
        };

        check_round(one, 0, false, 1.0, 1.0f16.next_up());
        check_round(max, 15, true, f16::MAX, f16::INFINITY);
        check_round(
            min_norm,
            -14,
            false,
            f16::MIN_POSITIVE_NORMAL,
            f16::MIN_POSITIVE_NORMAL.next_up(),
        );
        check_round(
            max_sub,
            -14,
            true,
            f16::MIN_POSITIVE_NORMAL.next_down(),
            f16::MIN_POSITIVE_NORMAL,
        );
        check_round(
            min_sub,
            -14,
            true,
            f16::MIN_POSITIVE_SUBNORMAL,
            f16::MIN_POSITIVE_SUBNORMAL.next_up(),
        );
        check_round(0, -14, false, 0.0, f16::MIN_POSITIVE_SUBNORMAL);
    }
}
