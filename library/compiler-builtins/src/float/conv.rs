use float::Float;
use int::{Int, CastInto};

fn int_to_float<I: Int, F: Float>(i: I) -> F where
    F::Int: CastInto<u32>,
    F::Int: CastInto<I>,
    I::UnsignedInt: CastInto<F::Int>,
    u32: CastInto<F::Int>,
{
    if i == I::ZERO {
        return F::ZERO;
    }

    let two = I::UnsignedInt::ONE + I::UnsignedInt::ONE;
    let four = two + two;
    let mant_dig = F::SIGNIFICAND_BITS + 1;
    let exponent_bias = F::EXPONENT_BIAS;

    let n = I::BITS;
    let (s, a) = i.extract_sign();
    let mut a = a;

    // number of significant digits
    let sd = n - a.leading_zeros();

    // exponent
    let mut e = sd - 1;

    if I::BITS < mant_dig {
        return F::from_parts(s,
            (e + exponent_bias).cast(),
            a.cast() << (mant_dig - e - 1));
    }

    a = if sd > mant_dig {
        /* start:  0000000000000000000001xxxxxxxxxxxxxxxxxxxxxxPQxxxxxxxxxxxxxxxxxx
        *  finish: 000000000000000000000000000000000000001xxxxxxxxxxxxxxxxxxxxxxPQR
        *                                                12345678901234567890123456
        *  1 = msb 1 bit
        *  P = bit MANT_DIG-1 bits to the right of 1
        *  Q = bit MANT_DIG bits to the right of 1
        *  R = "or" of all bits to the right of Q
        */
        let mant_dig_plus_one = mant_dig + 1;
        let mant_dig_plus_two = mant_dig + 2;
        a = if sd == mant_dig_plus_one {
            a << 1
        } else if sd == mant_dig_plus_two {
            a
        } else {
            (a >> (sd - mant_dig_plus_two)) |
            Int::from_bool((a & I::UnsignedInt::max_value()).wrapping_shl((n + mant_dig_plus_two) - sd) != Int::ZERO)
        };

        /* finish: */
        a |= Int::from_bool((a & four) != I::UnsignedInt::ZERO); /* Or P into R */
        a += Int::ONE; /* round - this step may add a significant bit */
        a >>= 2; /* dump Q and R */

        /* a is now rounded to mant_dig or mant_dig+1 bits */
        if (a & (I::UnsignedInt::ONE << mant_dig)) != Int::ZERO {
            a >>= 1; e += 1;
        }
        a
        /* a is now rounded to mant_dig bits */
    } else {
        a.wrapping_shl(mant_dig - sd)
        /* a is now rounded to mant_dig bits */
    };

    F::from_parts(s,
                 (e + exponent_bias).cast(),
                 a.cast())
}

intrinsics! {
    #[arm_aeabi_alias = __aeabi_i2f]
    pub extern "C" fn __floatsisf(i: i32) -> f32 {
        int_to_float(i)
    }

    #[arm_aeabi_alias = __aeabi_i2d]
    pub extern "C" fn __floatsidf(i: i32) -> f64 {
        int_to_float(i)
    }

    #[use_c_shim_if(all(target_arch = "x86", not(target_env = "msvc")))]
    #[arm_aeabi_alias = __aeabi_l2d]
    pub extern "C" fn __floatdidf(i: i64) -> f64 {
        // On x86_64 LLVM will use native instructions for this conversion, we
        // can just do it directly
        if cfg!(target_arch = "x86_64") {
            i as f64
        } else {
            int_to_float(i)
        }
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        int_to_float(i)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floattidf(i: i128) -> f64 {
        int_to_float(i)
    }

    #[arm_aeabi_alias = __aeabi_ui2f]
    pub extern "C" fn __floatunsisf(i: u32) -> f32 {
        int_to_float(i)
    }

    #[arm_aeabi_alias = __aeabi_ui2d]
    pub extern "C" fn __floatunsidf(i: u32) -> f64 {
        int_to_float(i)
    }

    #[use_c_shim_if(all(not(target_env = "msvc"),
                        any(target_arch = "x86",
                            all(not(windows), target_arch = "x86_64"))))]
    #[arm_aeabi_alias = __aeabi_ul2d]
    pub extern "C" fn __floatundidf(i: u64) -> f64 {
        int_to_float(i)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floatuntisf(i: u128) -> f32 {
        int_to_float(i)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floatuntidf(i: u128) -> f64 {
        int_to_float(i)
    }
}

#[derive(PartialEq)]
enum Sign {
    Positive,
    Negative
}

fn float_to_int<F: Float, I: Int>(f: F) -> I where
    F::Int: CastInto<u32>,
    F::Int: CastInto<I>,
{
    let f = f;
    let fixint_min = I::min_value();
    let fixint_max = I::max_value();
    let fixint_bits = I::BITS;
    let fixint_unsigned = fixint_min == I::ZERO;

    let sign_bit = F::SIGN_MASK;
    let significand_bits = F::SIGNIFICAND_BITS;
    let exponent_bias = F::EXPONENT_BIAS;
    //let exponent_max = F::exponent_max() as usize;

    // Break a into sign, exponent, significand
    let a_rep = F::repr(f);
    let a_abs = a_rep & !sign_bit;

    // this is used to work around -1 not being available for unsigned
    let sign = if (a_rep & sign_bit) == F::Int::ZERO { Sign::Positive } else { Sign::Negative };
    let mut exponent: u32 = (a_abs >> significand_bits).cast();
    let significand = (a_abs & F::SIGNIFICAND_MASK) | F::IMPLICIT_BIT;

    // if < 1 or unsigned & negative
    if exponent < exponent_bias ||
        fixint_unsigned && sign == Sign::Negative {
        return I::ZERO;
    }
    exponent -= exponent_bias;

    // If the value is infinity, saturate.
    // If the value is too large for the integer type, 0.
    if exponent >= (if fixint_unsigned {fixint_bits} else {fixint_bits -1}) {
        return if sign == Sign::Positive {fixint_max} else {fixint_min}
    }
    // If 0 <= exponent < significand_bits, right shift to get the result.
    // Otherwise, shift left.
    // (sign - 1) will never overflow as negative signs are already returned as 0 for unsigned
    let r: I = if exponent < significand_bits {
        (significand >> (significand_bits - exponent)).cast()
    } else {
        (significand << (exponent - significand_bits)).cast()
    };

    if sign == Sign::Negative {
        (!r).wrapping_add(I::ONE)
    } else {
        r
    }
}

intrinsics! {
    #[arm_aeabi_alias = __aeabi_f2iz]
    pub extern "C" fn __fixsfsi(f: f32) -> i32 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_f2lz]
    pub extern "C" fn __fixsfdi(f: f32) -> i64 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixsfti(f: f32) -> i128 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2iz]
    pub extern "C" fn __fixdfsi(f: f64) -> i32 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2lz]
    pub extern "C" fn __fixdfdi(f: f64) -> i64 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixdfti(f: f64) -> i128 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_f2uiz]
    pub extern "C" fn __fixunssfsi(f: f32) -> u32 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_f2ulz]
    pub extern "C" fn __fixunssfdi(f: f32) -> u64 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixunssfti(f: f32) -> u128 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2uiz]
    pub extern "C" fn __fixunsdfsi(f: f64) -> u32 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2ulz]
    pub extern "C" fn __fixunsdfdi(f: f64) -> u64 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixunsdfti(f: f64) -> u128 {
        float_to_int(f)
    }
}
