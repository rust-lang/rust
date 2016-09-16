use core::mem;
use core::num::Wrapping;

use float::Float;

macro_rules! add {
    ($intrinsic:ident: $ty:ty) => {
        /// Returns `a + b`
        #[allow(unused_parens)]
        #[cfg_attr(not(test), no_mangle)]
        pub extern fn $intrinsic(a: $ty, b: $ty) -> $ty {
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
                    return (<$ty as Float>::from_repr((a_abs | quiet_bit).0));
                }
                // anything + NaN = qNaN
                if b_abs > inf_rep {
                    return (<$ty as Float>::from_repr((b_abs | quiet_bit).0));
                }

                if a_abs == inf_rep {
                    // +/-infinity + -/+infinity = qNaN
                    if (a.repr() ^ b.repr()) == sign_bit.0 {
                        return (<$ty as Float>::from_repr(qnan_rep.0));
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
                        return (<$ty as Float>::from_repr(a.repr() & b.repr()));
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
                    return (<$ty as Float>::from_repr(0)); 
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
                return (<$ty>::from_repr((inf_rep | result_sign).0));
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
            return (<$ty>::from_repr(result.0));
        }
    }
}

add!(__addsf3: f32);
add!(__adddf3: f64);

// FIXME: Implement these using aliases
#[cfg(target_arch = "arm")]
#[cfg_attr(not(test), no_mangle)]
pub extern fn __aeabi_dadd(a: f64, b: f64) -> f64 {
    __adddf3(a, b)
}

#[cfg(target_arch = "arm")]
#[cfg_attr(not(test), no_mangle)]
pub extern fn __aeabi_fadd(a: f32, b: f32) -> f32 {
    __addsf3(a, b)
}

#[cfg(test)]
mod tests {
    use core::{f32, f64};

    use gcc_s;
    use qc::{U32, U64};
    use float::Float;

    // NOTE The tests below have special handing for NaN values.
    // Because NaN != NaN, the floating-point representations must be used
    // Because there are many diffferent values of NaN, and the implementation
    // doesn't care about calculating the 'correct' one, if both values are NaN
    // the values are considered equivalent.

    // TODO: Add F32/F64 to qc so that they print the right values (at the very least)
    quickcheck! {
        fn addsf3(a: U32, b: U32) -> bool {
            let (a, b) = (f32::from_repr(a.0), f32::from_repr(b.0));
            let x = super::__addsf3(a, b);

            if let Some(addsf3) = gcc_s::addsf3() {
               x.eq_repr(unsafe { addsf3(a, b) })
            } else {
                x.eq_repr(a + b)
            }
        }

        fn adddf3(a: U64, b: U64) -> bool {
            let (a, b) = (f64::from_repr(a.0), f64::from_repr(b.0));
            let x = super::__adddf3(a, b);

            if let Some(adddf3) = gcc_s::adddf3() {
                x.eq_repr(unsafe { adddf3(a, b) })
            } else {
                x.eq_repr(a + b)
            }
        }
    }
    
    // More tests for special float values

    #[test]
    fn test_float_tiny_plus_tiny() {
        let tiny = f32::from_repr(1);
        let r = super::__addsf3(tiny, tiny);
        assert!(r.eq_repr(tiny + tiny));
    }

    #[test]
    fn test_double_tiny_plus_tiny() {
        let tiny = f64::from_repr(1);
        let r = super::__adddf3(tiny, tiny);
        assert!(r.eq_repr(tiny + tiny));
    }

    #[test]
    fn test_float_small_plus_small() {
        let a = f32::from_repr(327);
        let b = f32::from_repr(256);
        let r = super::__addsf3(a, b);
        assert!(r.eq_repr(a + b));
    }

    #[test]
    fn test_double_small_plus_small() {
        let a = f64::from_repr(327);
        let b = f64::from_repr(256);
        let r = super::__adddf3(a, b);
        assert!(r.eq_repr(a + b));
    }

    #[test]
    fn test_float_one_plus_one() {
        let r = super::__addsf3(1f32, 1f32);
        assert!(r.eq_repr(1f32 + 1f32));
    }

    #[test]
    fn test_double_one_plus_one() {
        let r = super::__adddf3(1f64, 1f64);
        assert!(r.eq_repr(1f64 + 1f64));
    }

    #[test]
    fn test_float_different_nan() {
        let a = f32::from_repr(1);
        let b = f32::from_repr(0b11111111100100010001001010101010);
        let x = super::__addsf3(a, b);
        let y = a + b;
        assert!(x.eq_repr(y));
    }

    #[test]
    fn test_double_different_nan() {
        let a = f64::from_repr(1);
        let b = f64::from_repr(
            0b1111111111110010001000100101010101001000101010000110100011101011);
        let x = super::__adddf3(a, b);
        let y = a + b;
        assert!(x.eq_repr(y));
    }

    #[test]
    fn test_float_nan() {
        let r = super::__addsf3(f32::NAN, 1.23);
        assert_eq!(r.repr(), f32::NAN.repr());
    }

    #[test]
    fn test_double_nan() {
        let r = super::__adddf3(f64::NAN, 1.23);
        assert_eq!(r.repr(), f64::NAN.repr());
    }

    #[test]
    fn test_float_inf() {
        let r = super::__addsf3(f32::INFINITY, -123.4);
        assert_eq!(r, f32::INFINITY);
    }

    #[test]
    fn test_double_inf() {
        let r = super::__adddf3(f64::INFINITY, -123.4);
        assert_eq!(r, f64::INFINITY);
    }
}
