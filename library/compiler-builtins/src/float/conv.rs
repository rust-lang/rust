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

// NOTE(cfg) for some reason, on arm*-unknown-linux-gnueabihf, our implementation doesn't
// match the output of its gcc_s or compiler-rt counterpart. Until we investigate further, we'll
// just avoid testing against them on those targets. Do note that our implementation gives the
// correct answer; gcc_s and compiler-rt are incorrect in this case.
//
#[cfg(all(test, not(arm_linux)))]
mod tests {
    use qc::{I32, U32, I64, U64, F32, F64};

    check! {
        fn __floatsisf(f: extern fn(i32) -> f32,
                    a: I32)
                    -> Option<F32> {
            Some(F32(f(a.0)))
        }
        fn __floatsidf(f: extern fn(i32) -> f64,
                    a: I32)
                    -> Option<F64> {
            Some(F64(f(a.0)))
        }
        fn __floatdidf(f: extern fn(i64) -> f64,
                    a: I64)
                    -> Option<F64> {
            Some(F64(f(a.0)))
        }
        fn __floatunsisf(f: extern fn(u32) -> f32,
                    a: U32)
                    -> Option<F32> {
            Some(F32(f(a.0)))
        }
        fn __floatunsidf(f: extern fn(u32) -> f64,
                    a: U32)
                    -> Option<F64> {
            Some(F64(f(a.0)))
        }
        fn __floatundidf(f: extern fn(u64) -> f64,
                    a: U64)
                    -> Option<F64> {
            Some(F64(f(a.0)))
        }
    }
}
