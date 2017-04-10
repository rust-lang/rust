use float::Float;
use int::Int;

macro_rules! fp_overflow {
    (infinity, $fty:ty, $sign: expr) => {
        return {
            <$fty as Float>::from_parts(
                $sign,
                <$fty as Float>::exponent_max() as <$fty as Float>::Int,
                0 as <$fty as Float>::Int)
        }
    }
}

macro_rules! fp_convert {
    ($intrinsic:ident: $ity:ty, $fty:ty) => {

    pub extern "C" fn $intrinsic(i: $ity) -> $fty {
        if i == 0 {
            return 0.0
        }

        let mant_dig = <$fty>::significand_bits() + 1;
        let exponent_bias = <$fty>::exponent_bias();

        let n = <$ity>::bits();
        let (s, a) = i.extract_sign();
        let mut a = a;

        // number of significant digits
        let sd = n - a.leading_zeros();

        // exponent
        let mut e = sd - 1;

        if <$ity>::bits() < mant_dig {
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
    }
    }
}

fp_convert!(__floatsisf: i32, f32);
fp_convert!(__floatsidf: i32, f64);
fp_convert!(__floatdidf: i64, f64);
fp_convert!(__floatunsisf: u32, f32);
fp_convert!(__floatunsidf: u32, f64);
fp_convert!(__floatundidf: u64, f64);

#[derive(PartialEq, Debug)]
enum Sign {
    Positive,
    Negative
}

macro_rules! fp_fix {
    ($intrinsic:ident: $fty:ty, $ity:ty) => {
        pub extern "C" fn $intrinsic(f: $fty) -> $ity {
            let fixint_min = <$ity>::min_value();
            let fixint_max = <$ity>::max_value();
            let fixint_bits = <$ity>::bits() as usize;
            let fixint_unsigned = fixint_min == 0;

            let sign_bit = <$fty>::sign_mask();
            let significand_bits = <$fty>::significand_bits() as usize;
            let exponent_bias = <$fty>::exponent_bias() as usize;
            //let exponent_max = <$fty>::exponent_max() as usize;

            // Break a into sign, exponent, significand
            let a_rep = <$fty>::repr(f);
            let a_abs = a_rep & !sign_bit;

            // this is used to work around -1 not being available for unsigned
            let sign = if (a_rep & sign_bit) == 0 { Sign::Positive } else { Sign::Negative };
            let mut exponent = (a_abs >> significand_bits) as usize;
            let significand = (a_abs & <$fty>::significand_mask()) | <$fty>::implicit_bit();

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
        }
    }
}

fp_fix!(__fixsfsi: f32, i32);
fp_fix!(__fixsfdi: f32, i64);
fp_fix!(__fixdfsi: f64, i32);
fp_fix!(__fixdfdi: f64, i64);

fp_fix!(__fixunssfsi: f32, u32);
fp_fix!(__fixunssfdi: f32, u64);
fp_fix!(__fixunsdfsi: f64, u32);
fp_fix!(__fixunsdfdi: f64, u64);
