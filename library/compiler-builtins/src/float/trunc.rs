use float::Float;
use int::{CastInto, Int};

fn trunc<F: Float, R: Float>(a: F) -> R
where
    F::Int: CastInto<u64>,
    F::Int: CastInto<u32>,
    u64: CastInto<F::Int>,
    u32: CastInto<F::Int>,

    R::Int: CastInto<u32>,
    u32: CastInto<R::Int>,
    F::Int: CastInto<R::Int>,
{
    let src_zero = F::Int::ZERO;
    let src_one = F::Int::ONE;
    let src_bits = F::BITS;
    let src_exp_bias = F::EXPONENT_BIAS;

    let src_min_normal = F::IMPLICIT_BIT;
    let src_significand_mask = F::SIGNIFICAND_MASK;
    let src_infinity = F::EXPONENT_MASK;
    let src_sign_mask = F::SIGN_MASK;
    let src_abs_mask = src_sign_mask - src_one;
    let round_mask = (src_one << (F::SIGNIFICAND_BITS - R::SIGNIFICAND_BITS)) - src_one;
    let halfway = src_one << (F::SIGNIFICAND_BITS - R::SIGNIFICAND_BITS - 1);
    let src_qnan = src_one << (F::SIGNIFICAND_BITS - 1);
    let src_nan_code = src_qnan - src_one;

    let dst_zero = R::Int::ZERO;
    let dst_one = R::Int::ONE;
    let dst_bits = R::BITS;
    let dst_inf_exp = R::EXPONENT_MAX;
    let dst_exp_bias = R::EXPONENT_BIAS;

    let underflow_exponent: F::Int = (src_exp_bias + 1 - dst_exp_bias).cast();
    let overflow_exponent: F::Int = (src_exp_bias + dst_inf_exp - dst_exp_bias).cast();
    let underflow: F::Int = underflow_exponent << F::SIGNIFICAND_BITS;
    let overflow: F::Int = overflow_exponent << F::SIGNIFICAND_BITS;

    let dst_qnan = R::Int::ONE << (R::SIGNIFICAND_BITS - 1);
    let dst_nan_code = dst_qnan - dst_one;

    let sign_bits_delta = F::SIGNIFICAND_BITS - R::SIGNIFICAND_BITS;
    // Break a into a sign and representation of the absolute value.
    let a_abs = a.repr() & src_abs_mask;
    let sign = a.repr() & src_sign_mask;
    let mut abs_result: R::Int;

    if a_abs.wrapping_sub(underflow) < a_abs.wrapping_sub(overflow) {
        // The exponent of a is within the range of normal numbers in the
        // destination format.  We can convert by simply right-shifting with
        // rounding and adjusting the exponent.
        abs_result = (a_abs >> sign_bits_delta).cast();
        let tmp = src_exp_bias.wrapping_sub(dst_exp_bias) << R::SIGNIFICAND_BITS;
        abs_result = abs_result.wrapping_sub(tmp.cast());

        let round_bits = a_abs & round_mask;
        if round_bits > halfway {
            // Round to nearest.
            abs_result += dst_one;
        } else if round_bits == halfway {
            // Tie to even.
            abs_result += abs_result & dst_one;
        };
    } else if a_abs > src_infinity {
        // a is NaN.
        // Conjure the result by beginning with infinity, setting the qNaN
        // bit and inserting the (truncated) trailing NaN field.
        abs_result = (dst_inf_exp << R::SIGNIFICAND_BITS).cast();
        abs_result |= dst_qnan;
        abs_result |= dst_nan_code
            & ((a_abs & src_nan_code) >> (F::SIGNIFICAND_BITS - R::SIGNIFICAND_BITS)).cast();
    } else if a_abs >= overflow {
        // a overflows to infinity.
        abs_result = (dst_inf_exp << R::SIGNIFICAND_BITS).cast();
    } else {
        // a underflows on conversion to the destination type or is an exact
        // zero.  The result may be a denormal or zero.  Extract the exponent
        // to get the shift amount for the denormalization.
        let a_exp: u32 = (a_abs >> F::SIGNIFICAND_BITS).cast();
        let shift = src_exp_bias - dst_exp_bias - a_exp + 1;

        let significand = (a.repr() & src_significand_mask) | src_min_normal;

        // Right shift by the denormalization amount with sticky.
        if shift > F::SIGNIFICAND_BITS {
            abs_result = dst_zero;
        } else {
            let sticky = if (significand << (src_bits - shift)) != src_zero {
                src_one
            } else {
                src_zero
            };
            let denormalized_significand: F::Int = significand >> shift | sticky;
            abs_result =
                (denormalized_significand >> (F::SIGNIFICAND_BITS - R::SIGNIFICAND_BITS)).cast();
            let round_bits = denormalized_significand & round_mask;
            // Round to nearest
            if round_bits > halfway {
                abs_result += dst_one;
            }
            // Ties to even
            else if round_bits == halfway {
                abs_result += abs_result & dst_one;
            };
        }
    }

    // Apply the signbit to the absolute value.
    R::from_repr(abs_result | sign.wrapping_shr(src_bits - dst_bits).cast())
}

intrinsics! {
    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_d2f]
    pub extern "C" fn __truncdfsf2(a: f64) -> f32 {
        trunc(a)
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn __truncdfsf2vfp(a: f64) -> f32 {
        a as f32
    }
}
