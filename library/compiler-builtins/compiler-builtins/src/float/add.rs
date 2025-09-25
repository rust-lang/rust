use crate::float::Float;
use crate::int::{CastFrom, CastInto, Int, MinInt};

/// Returns `a + b`
fn add<F: Float>(a: F, b: F) -> F
where
    u32: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    i32: CastInto<F::Int>,
    F::Int: CastInto<i32>,
{
    let one = F::Int::ONE;
    let zero = F::Int::ZERO;

    let bits: F::Int = F::BITS.cast();
    let significand_bits = F::SIG_BITS;
    let max_exponent = F::EXP_SAT;

    let implicit_bit = F::IMPLICIT_BIT;
    let significand_mask = F::SIG_MASK;
    let sign_bit = F::SIGN_MASK as F::Int;
    let abs_mask = sign_bit - one;
    let exponent_mask = F::EXP_MASK;
    let inf_rep = exponent_mask;
    let quiet_bit = implicit_bit >> 1;
    let qnan_rep = exponent_mask | quiet_bit;

    let mut a_rep = a.to_bits();
    let mut b_rep = b.to_bits();
    let a_abs = a_rep & abs_mask;
    let b_abs = b_rep & abs_mask;

    // Detect if a or b is zero, infinity, or NaN.
    if a_abs.wrapping_sub(one) >= inf_rep - one || b_abs.wrapping_sub(one) >= inf_rep - one {
        // NaN + anything = qNaN
        if a_abs > inf_rep {
            return F::from_bits(a_abs | quiet_bit);
        }
        // anything + NaN = qNaN
        if b_abs > inf_rep {
            return F::from_bits(b_abs | quiet_bit);
        }

        if a_abs == inf_rep {
            // +/-infinity + -/+infinity = qNaN
            if (a.to_bits() ^ b.to_bits()) == sign_bit {
                return F::from_bits(qnan_rep);
            } else {
                // +/-infinity + anything remaining = +/- infinity
                return a;
            }
        }

        // anything remaining + +/-infinity = +/-infinity
        if b_abs == inf_rep {
            return b;
        }

        // zero + anything = anything
        if a_abs == MinInt::ZERO {
            // but we need to get the sign right for zero + zero
            if b_abs == MinInt::ZERO {
                return F::from_bits(a.to_bits() & b.to_bits());
            } else {
                return b;
            }
        }

        // anything + zero = anything
        if b_abs == MinInt::ZERO {
            return a;
        }
    }

    // Swap a and b if necessary so that a has the larger absolute value.
    if b_abs > a_abs {
        // Don't use mem::swap because it may generate references to memcpy in unoptimized code.
        let tmp = a_rep;
        a_rep = b_rep;
        b_rep = tmp;
    }

    // Extract the exponent and significand from the (possibly swapped) a and b.
    let mut a_exponent: i32 = ((a_rep & exponent_mask) >> significand_bits).cast();
    let mut b_exponent: i32 = ((b_rep & exponent_mask) >> significand_bits).cast();
    let mut a_significand = a_rep & significand_mask;
    let mut b_significand = b_rep & significand_mask;

    // normalize any denormals, and adjust the exponent accordingly.
    if a_exponent == 0 {
        let (exponent, significand) = F::normalize(a_significand);
        a_exponent = exponent;
        a_significand = significand;
    }
    if b_exponent == 0 {
        let (exponent, significand) = F::normalize(b_significand);
        b_exponent = exponent;
        b_significand = significand;
    }

    // The sign of the result is the sign of the larger operand, a.  If they
    // have opposite signs, we are performing a subtraction; otherwise addition.
    let result_sign = a_rep & sign_bit;
    let subtraction = ((a_rep ^ b_rep) & sign_bit) != zero;

    // Shift the significands to give us round, guard and sticky, and or in the
    // implicit significand bit.  (If we fell through from the denormal path it
    // was already set by normalize(), but setting it twice won't hurt
    // anything.)
    a_significand = (a_significand | implicit_bit) << 3;
    b_significand = (b_significand | implicit_bit) << 3;

    // Shift the significand of b by the difference in exponents, with a sticky
    // bottom bit to get rounding correct.
    let align = a_exponent.wrapping_sub(b_exponent).cast();
    if align != MinInt::ZERO {
        if align < bits {
            let sticky = F::Int::from_bool(
                b_significand << u32::cast_from(bits.wrapping_sub(align)) != MinInt::ZERO,
            );
            b_significand = (b_significand >> u32::cast_from(align)) | sticky;
        } else {
            b_significand = one; // sticky; b is known to be non-zero.
        }
    }
    if subtraction {
        a_significand = a_significand.wrapping_sub(b_significand);
        // If a == -b, return +zero.
        if a_significand == MinInt::ZERO {
            return F::from_bits(MinInt::ZERO);
        }

        // If partial cancellation occurred, we need to left-shift the result
        // and adjust the exponent:
        if a_significand < implicit_bit << 3 {
            let shift = a_significand.leading_zeros() as i32
                - (implicit_bit << 3u32).leading_zeros() as i32;
            a_significand <<= shift;
            a_exponent -= shift;
        }
    } else {
        // addition
        a_significand += b_significand;

        // If the addition carried up, we need to right-shift the result and
        // adjust the exponent:
        if a_significand & (implicit_bit << 4) != MinInt::ZERO {
            let sticky = F::Int::from_bool(a_significand & one != MinInt::ZERO);
            a_significand = (a_significand >> 1) | sticky;
            a_exponent += 1;
        }
    }

    // If we have overflowed the type, return +/- infinity:
    if a_exponent >= max_exponent as i32 {
        return F::from_bits(inf_rep | result_sign);
    }

    if a_exponent <= 0 {
        // Result is denormal before rounding; the exponent is zero and we
        // need to shift the significand.
        let shift = (1 - a_exponent).cast();
        let sticky = F::Int::from_bool(
            (a_significand << u32::cast_from(bits.wrapping_sub(shift))) != MinInt::ZERO,
        );
        a_significand = (a_significand >> u32::cast_from(shift)) | sticky;
        a_exponent = 0;
    }

    // Low three bits are round, guard, and sticky.
    let a_significand_i32: i32 = a_significand.cast_lossy();
    let round_guard_sticky: i32 = a_significand_i32 & 0x7;

    // Shift the significand into place, and mask off the implicit bit.
    let mut result = (a_significand >> 3) & significand_mask;

    // Insert the exponent and sign.
    result |= a_exponent.cast() << significand_bits;
    result |= result_sign;

    // Final rounding.  The result may overflow to infinity, but that is the
    // correct result in that case.
    if round_guard_sticky > 0x4 {
        result += one;
    }
    if round_guard_sticky == 0x4 {
        result += result & one;
    }

    F::from_bits(result)
}

intrinsics! {
    #[cfg(f16_enabled)]
    pub extern "C" fn __addhf3(a: f16, b: f16) -> f16 {
        add(a, b)
    }

    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_fadd]
    pub extern "C" fn __addsf3(a: f32, b: f32) -> f32 {
        add(a, b)
    }

    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_dadd]
    pub extern "C" fn __adddf3(a: f64, b: f64) -> f64 {
        add(a, b)
    }

    #[ppc_alias = __addkf3]
    #[cfg(f128_enabled)]
    pub extern "C" fn __addtf3(a: f128, b: f128) -> f128 {
        add(a, b)
    }
}
