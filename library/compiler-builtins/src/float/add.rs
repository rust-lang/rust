use core::mem;
use core::num::Wrapping;

use float::Float;

/// Returns `a + b`
macro_rules! add {
    ($a:expr, $b:expr, $ty:ty) => ({
        let a = $a;
        let b = $b;
        let one = Wrapping(1 as <$ty as Float>::Int);
        let zero = Wrapping(0 as <$ty as Float>::Int);

        let bits =             Wrapping(<$ty>::bits() as <$ty as Float>::Int);
        let significand_bits = Wrapping(<$ty>::significand_bits() as <$ty as Float>::Int);
        let exponent_bits =    bits - significand_bits - one;
        let max_exponent =     (one << exponent_bits.0 as usize) - one;

        let implicit_bit =     one << significand_bits.0 as usize;
        let significand_mask = implicit_bit - one;
        let sign_bit =         one << (significand_bits + exponent_bits).0 as usize;
        let abs_mask =         sign_bit - one;
        let exponent_mask =    abs_mask ^ significand_mask;
        let inf_rep =          exponent_mask;
        let quiet_bit =        implicit_bit >> 1;
        let qnan_rep =         exponent_mask | quiet_bit;

        let mut a_rep = Wrapping(a.repr());
        let mut b_rep = Wrapping(b.repr());
        let a_abs = a_rep & abs_mask;
        let b_abs = b_rep & abs_mask;

        // Detect if a or b is zero, infinity, or NaN.
        if a_abs - one >= inf_rep - one ||
            b_abs - one >= inf_rep - one {
            // NaN + anything = qNaN
            if a_abs > inf_rep {
                return <$ty as Float>::from_repr((a_abs | quiet_bit).0);
            }
            // anything + NaN = qNaN
            if b_abs > inf_rep {
                return <$ty as Float>::from_repr((b_abs | quiet_bit).0);
            }

            if a_abs == inf_rep {
                // +/-infinity + -/+infinity = qNaN
                if (a.repr() ^ b.repr()) == sign_bit.0 {
                    return <$ty as Float>::from_repr(qnan_rep.0);
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
            if a_abs.0 == 0 {
                // but we need to get the sign right for zero + zero
                if b_abs.0 == 0 {
                    return <$ty as Float>::from_repr(a.repr() & b.repr());
                } else {
                    return b;
                }
            }

            // anything + zero = anything
            if b_abs.0 == 0 {
                 return a;
            }
        }

        // Swap a and b if necessary so that a has the larger absolute value.
        if b_abs > a_abs {
            mem::swap(&mut a_rep, &mut b_rep);
        }

        // Extract the exponent and significand from the (possibly swapped) a and b.
        let mut a_exponent = Wrapping((a_rep >> significand_bits.0 as usize & max_exponent).0 as i32);
        let mut b_exponent = Wrapping((b_rep >> significand_bits.0 as usize & max_exponent).0 as i32);
        let mut a_significand = a_rep & significand_mask;
        let mut b_significand = b_rep & significand_mask;

        // normalize any denormals, and adjust the exponent accordingly.
        if a_exponent.0 == 0 {
            let (exponent, significand) = <$ty>::normalize(a_significand.0);
            a_exponent = Wrapping(exponent);
            a_significand = Wrapping(significand);
        }
        if b_exponent.0 == 0 {
            let (exponent, significand) = <$ty>::normalize(b_significand.0);
            b_exponent = Wrapping(exponent);
            b_significand = Wrapping(significand);
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
        let align = Wrapping((a_exponent - b_exponent).0 as <$ty as Float>::Int);
        if align.0 != 0 {
            if align < bits {
                let sticky = ((b_significand << (bits - align).0 as usize).0 != 0) as <$ty as Float>::Int;
                b_significand = (b_significand >> align.0 as usize) | Wrapping(sticky);
            } else {
                b_significand = one; // sticky; b is known to be non-zero.
            }
        }
        if subtraction {
            a_significand -= b_significand;
            // If a == -b, return +zero.
            if a_significand.0 == 0 {
                return <$ty as Float>::from_repr(0);
            }

            // If partial cancellation occured, we need to left-shift the result
            // and adjust the exponent:
            if a_significand < implicit_bit << 3 {
                let shift = a_significand.0.leading_zeros() as i32
                    - (implicit_bit << 3).0.leading_zeros() as i32;
                a_significand <<= shift as usize;
                a_exponent -= Wrapping(shift);
            }
        } else /* addition */ {
            a_significand += b_significand;

            // If the addition carried up, we need to right-shift the result and
            // adjust the exponent:
            if (a_significand & implicit_bit << 4).0 != 0 {
                let sticky = ((a_significand & one).0 != 0) as <$ty as Float>::Int;
                a_significand = a_significand >> 1 | Wrapping(sticky);
                a_exponent += Wrapping(1);
            }
        }

        // If we have overflowed the type, return +/- infinity:
        if a_exponent >= Wrapping(max_exponent.0 as i32) {
            return <$ty>::from_repr((inf_rep | result_sign).0);
        }

        if a_exponent.0 <= 0 {
            // Result is denormal before rounding; the exponent is zero and we
            // need to shift the significand.
            let shift = Wrapping((Wrapping(1) - a_exponent).0 as <$ty as Float>::Int);
            let sticky = ((a_significand << (bits - shift).0 as usize).0 != 0) as <$ty as Float>::Int;
            a_significand = a_significand >> shift.0 as usize | Wrapping(sticky);
            a_exponent = Wrapping(0);
        }

        // Low three bits are round, guard, and sticky.
        let round_guard_sticky: i32 = (a_significand.0 & 0x7) as i32;

        // Shift the significand into place, and mask off the implicit bit.
        let mut result = a_significand >> 3 & significand_mask;

        // Insert the exponent and sign.
        result |= Wrapping(a_exponent.0 as <$ty as Float>::Int) << significand_bits.0 as usize;
        result |= result_sign;

        // Final rounding.  The result may overflow to infinity, but that is the
        // correct result in that case.
        if round_guard_sticky > 0x4 { result += one; }
        if round_guard_sticky == 0x4 { result += result & one; }

        <$ty>::from_repr(result.0)
    })
}

intrinsics! {
    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_fadd]
    pub extern "C" fn __addsf3(a: f32, b: f32) -> f32 {
        add!(a, b, f32)
    }

    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_dadd]
    pub extern "C" fn __adddf3(a: f64, b: f64) -> f64 {
        add!(a, b, f64)
    }
}
