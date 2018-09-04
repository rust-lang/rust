use float::Float;
use int::Int;

macro_rules! int_to_float {
    ($i:expr, $ity:ty, $fty:ty) => ({
        let i = $i;
        if i == 0 {
            return 0.0
        }

        let mant_dig = <$fty>::SIGNIFICAND_BITS + 1;
        let exponent_bias = <$fty>::EXPONENT_BIAS;

        let n = <$ity>::BITS;
        let (s, a) = i.extract_sign();
        let mut a = a;

        // number of significant digits
        let sd = n - a.leading_zeros();

        // exponent
        let mut e = sd - 1;

        if <$ity>::BITS < mant_dig {
            return <$fty>::from_parts(s,
                (e + exponent_bias) as <$fty as Float>::Int,
                (a as <$fty as Float>::Int) << (mant_dig - e - 1))
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
                (a >> (sd - mant_dig_plus_two)) as <$ity as Int>::UnsignedInt |
                ((a & <$ity as Int>::UnsignedInt::max_value()).wrapping_shl((n + mant_dig_plus_two) - sd) != 0) as <$ity as Int>::UnsignedInt
            };

            /* finish: */
            a |= ((a & 4) != 0) as <$ity as Int>::UnsignedInt; /* Or P into R */
            a += 1; /* round - this step may add a significant bit */
            a >>= 2; /* dump Q and R */

            /* a is now rounded to mant_dig or mant_dig+1 bits */
            if (a & (1 << mant_dig)) != 0 {
                a >>= 1; e += 1;
            }
            a
            /* a is now rounded to mant_dig bits */
        } else {
            a.wrapping_shl(mant_dig - sd)
            /* a is now rounded to mant_dig bits */
        };

        <$fty>::from_parts(s,
            (e + exponent_bias) as <$fty as Float>::Int,
            a as <$fty as Float>::Int)
    })
}

intrinsics! {
    #[arm_aeabi_alias = __aeabi_i2f]
    pub extern "C" fn __floatsisf(i: i32) -> f32 {
        int_to_float!(i, i32, f32)
    }

    #[arm_aeabi_alias = __aeabi_i2d]
    pub extern "C" fn __floatsidf(i: i32) -> f64 {
        int_to_float!(i, i32, f64)
    }

    #[use_c_shim_if(any(
        all(target_arch = "x86", not(target_env = "msvc")),
        all(target_arch = "x86_64", not(windows)),
    ))]
    #[arm_aeabi_alias = __aeabi_l2f]
    pub extern "C" fn __floatdisf(i: i64) -> f32 {
        // On x86_64 LLVM will use native instructions for this conversion, we
        // can just do it directly
        if cfg!(target_arch = "x86_64") {
            i as f32
        } else {
            int_to_float!(i, i64, f32)
        }
    }

    #[use_c_shim_if(all(target_arch = "x86", not(target_env = "msvc")))]
    #[arm_aeabi_alias = __aeabi_l2d]
    pub extern "C" fn __floatdidf(i: i64) -> f64 {
        // On x86_64 LLVM will use native instructions for this conversion, we
        // can just do it directly
        if cfg!(target_arch = "x86_64") {
            i as f64
        } else {
            int_to_float!(i, i64, f64)
        }
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        int_to_float!(i, i128, f32)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floattidf(i: i128) -> f64 {
        int_to_float!(i, i128, f64)
    }

    #[arm_aeabi_alias = __aeabi_ui2f]
    pub extern "C" fn __floatunsisf(i: u32) -> f32 {
        int_to_float!(i, u32, f32)
    }

    #[arm_aeabi_alias = __aeabi_ui2d]
    pub extern "C" fn __floatunsidf(i: u32) -> f64 {
        int_to_float!(i, u32, f64)
    }

    #[use_c_shim_if(any(
        all(target_arch = "x86", not(target_env = "msvc")),
        all(target_arch = "x86_64", not(windows)),
    ))]
    #[arm_aeabi_alias = __aeabi_ul2f]
    pub extern "C" fn __floatundisf(i: u64) -> f32 {
        int_to_float!(i, u64, f32)
    }

    #[use_c_shim_if(any(
        all(target_arch = "x86", not(target_env = "msvc")),
        all(target_arch = "x86_64", not(windows)),
    ))]
    #[arm_aeabi_alias = __aeabi_ul2d]
    pub extern "C" fn __floatundidf(i: u64) -> f64 {
        int_to_float!(i, u64, f64)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floatuntisf(i: u128) -> f32 {
        int_to_float!(i, u128, f32)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floatuntidf(i: u128) -> f64 {
        int_to_float!(i, u128, f64)
    }
}

#[derive(PartialEq)]
enum Sign {
    Positive,
    Negative
}

macro_rules! float_to_int {
    ($f:expr, $fty:ty, $ity:ty) => ({
        let f = $f;
        let fixint_min = <$ity>::min_value();
        let fixint_max = <$ity>::max_value();
        let fixint_bits = <$ity>::BITS as usize;
        let fixint_unsigned = fixint_min == 0;

        let sign_bit = <$fty>::SIGN_MASK;
        let significand_bits = <$fty>::SIGNIFICAND_BITS as usize;
        let exponent_bias = <$fty>::EXPONENT_BIAS as usize;
        //let exponent_max = <$fty>::exponent_max() as usize;

        // Break a into sign, exponent, significand
        let a_rep = <$fty>::repr(f);
        let a_abs = a_rep & !sign_bit;

        // this is used to work around -1 not being available for unsigned
        let sign = if (a_rep & sign_bit) == 0 { Sign::Positive } else { Sign::Negative };
        let mut exponent = (a_abs >> significand_bits) as usize;
        let significand = (a_abs & <$fty>::SIGNIFICAND_MASK) | <$fty>::IMPLICIT_BIT;

        // if < 1 or unsigned & negative
        if  exponent < exponent_bias ||
            fixint_unsigned && sign == Sign::Negative {
            return 0
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
        let r = if exponent < significand_bits {
            (significand >> (significand_bits - exponent)) as $ity
        } else {
            (significand as $ity) << (exponent - significand_bits)
        };

        if sign == Sign::Negative {
            (!r).wrapping_add(1)
        } else {
            r
        }
    })
}

intrinsics! {
    #[arm_aeabi_alias = __aeabi_f2iz]
    pub extern "C" fn __fixsfsi(f: f32) -> i32 {
        float_to_int!(f, f32, i32)
    }

    #[arm_aeabi_alias = __aeabi_f2lz]
    pub extern "C" fn __fixsfdi(f: f32) -> i64 {
        float_to_int!(f, f32, i64)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixsfti(f: f32) -> i128 {
        float_to_int!(f, f32, i128)
    }

    #[arm_aeabi_alias = __aeabi_d2iz]
    pub extern "C" fn __fixdfsi(f: f64) -> i32 {
        float_to_int!(f, f64, i32)
    }

    #[arm_aeabi_alias = __aeabi_d2lz]
    pub extern "C" fn __fixdfdi(f: f64) -> i64 {
        float_to_int!(f, f64, i64)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixdfti(f: f64) -> i128 {
        float_to_int!(f, f64, i128)
    }

    #[arm_aeabi_alias = __aeabi_f2uiz]
    pub extern "C" fn __fixunssfsi(f: f32) -> u32 {
        float_to_int!(f, f32, u32)
    }

    #[arm_aeabi_alias = __aeabi_f2ulz]
    pub extern "C" fn __fixunssfdi(f: f32) -> u64 {
        float_to_int!(f, f32, u64)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixunssfti(f: f32) -> u128 {
        float_to_int!(f, f32, u128)
    }

    #[arm_aeabi_alias = __aeabi_d2uiz]
    pub extern "C" fn __fixunsdfsi(f: f64) -> u32 {
        float_to_int!(f, f64, u32)
    }

    #[arm_aeabi_alias = __aeabi_d2ulz]
    pub extern "C" fn __fixunsdfdi(f: f64) -> u64 {
        float_to_int!(f, f64, u64)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixunsdfti(f: f64) -> u128 {
        float_to_int!(f, f64, u128)
    }
}
