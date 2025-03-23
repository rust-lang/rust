use crate::float::Float;
use crate::int::{CastInto, Int, MinInt};

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
    let src_exp_bias = F::EXP_BIAS;

    let src_min_normal = F::IMPLICIT_BIT;
    let src_sig_mask = F::SIG_MASK;
    let src_infinity = F::EXP_MASK;
    let src_sign_mask = F::SIGN_MASK;
    let src_abs_mask = src_sign_mask - src_one;
    let round_mask = (src_one << (F::SIG_BITS - R::SIG_BITS)) - src_one;
    let halfway = src_one << (F::SIG_BITS - R::SIG_BITS - 1);
    let src_qnan = src_one << (F::SIG_BITS - 1);
    let src_nan_code = src_qnan - src_one;

    let dst_zero = R::Int::ZERO;
    let dst_one = R::Int::ONE;
    let dst_bits = R::BITS;
    let dst_inf_exp = R::EXP_SAT;
    let dst_exp_bias = R::EXP_BIAS;

    let underflow_exponent: F::Int = (src_exp_bias + 1 - dst_exp_bias).cast();
    let overflow_exponent: F::Int = (src_exp_bias + dst_inf_exp - dst_exp_bias).cast();
    let underflow: F::Int = underflow_exponent << F::SIG_BITS;
    let overflow: F::Int = overflow_exponent << F::SIG_BITS;

    let dst_qnan = R::Int::ONE << (R::SIG_BITS - 1);
    let dst_nan_code = dst_qnan - dst_one;

    let sig_bits_delta = F::SIG_BITS - R::SIG_BITS;
    // Break a into a sign and representation of the absolute value.
    let a_abs = a.to_bits() & src_abs_mask;
    let sign = a.to_bits() & src_sign_mask;
    let mut abs_result: R::Int;

    if a_abs.wrapping_sub(underflow) < a_abs.wrapping_sub(overflow) {
        // The exponent of a is within the range of normal numbers in the
        // destination format.  We can convert by simply right-shifting with
        // rounding and adjusting the exponent.
        abs_result = (a_abs >> sig_bits_delta).cast();
        // Cast before shifting to prevent overflow.
        let bias_diff: R::Int = src_exp_bias.wrapping_sub(dst_exp_bias).cast();
        let tmp = bias_diff << R::SIG_BITS;
        abs_result = abs_result.wrapping_sub(tmp);

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
        // Cast before shifting to prevent overflow.
        let dst_inf_exp: R::Int = dst_inf_exp.cast();
        abs_result = dst_inf_exp << R::SIG_BITS;
        abs_result |= dst_qnan;
        abs_result |= dst_nan_code & ((a_abs & src_nan_code) >> (F::SIG_BITS - R::SIG_BITS)).cast();
    } else if a_abs >= overflow {
        // a overflows to infinity.
        // Cast before shifting to prevent overflow.
        let dst_inf_exp: R::Int = dst_inf_exp.cast();
        abs_result = dst_inf_exp << R::SIG_BITS;
    } else {
        // a underflows on conversion to the destination type or is an exact
        // zero.  The result may be a denormal or zero.  Extract the exponent
        // to get the shift amount for the denormalization.
        let a_exp: u32 = (a_abs >> F::SIG_BITS).cast();
        let shift = src_exp_bias - dst_exp_bias - a_exp + 1;

        let significand = (a.to_bits() & src_sig_mask) | src_min_normal;

        // Right shift by the denormalization amount with sticky.
        if shift > F::SIG_BITS {
            abs_result = dst_zero;
        } else {
            let sticky = if (significand << (src_bits - shift)) != src_zero {
                src_one
            } else {
                src_zero
            };
            let denormalized_significand: F::Int = (significand >> shift) | sticky;
            abs_result = (denormalized_significand >> (F::SIG_BITS - R::SIG_BITS)).cast();
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
    R::from_bits(abs_result | sign.wrapping_shr(src_bits - dst_bits).cast())
}

intrinsics! {
    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_d2f]
    pub extern "C" fn __truncdfsf2(a: f64) -> f32 {
        trunc(a)
    }
}

intrinsics! {
    #[aapcs_on_arm]
    #[apple_f16_ret_abi]
    #[arm_aeabi_alias = __aeabi_f2h]
    #[cfg(f16_enabled)]
    pub extern "C" fn __truncsfhf2(a: f32) -> f16 {
        trunc(a)
    }

    #[aapcs_on_arm]
    #[apple_f16_ret_abi]
    #[cfg(f16_enabled)]
    pub extern "C" fn __gnu_f2h_ieee(a: f32) -> f16 {
        trunc(a)
    }

    #[aapcs_on_arm]
    #[apple_f16_ret_abi]
    #[arm_aeabi_alias = __aeabi_d2h]
    #[cfg(f16_enabled)]
    pub extern "C" fn __truncdfhf2(a: f64) -> f16 {
        trunc(a)
    }

    #[aapcs_on_arm]
    #[ppc_alias = __trunckfhf2]
    #[cfg(all(f16_enabled, f128_enabled))]
    pub extern "C" fn __trunctfhf2(a: f128) -> f16 {
        trunc(a)
    }

    #[aapcs_on_arm]
    #[ppc_alias = __trunckfsf2]
    #[cfg(f128_enabled)]
    pub extern "C" fn __trunctfsf2(a: f128) -> f32 {
        trunc(a)
    }

    #[aapcs_on_arm]
    #[ppc_alias = __trunckfdf2]
    #[cfg(f128_enabled)]
    pub extern "C" fn __trunctfdf2(a: f128) -> f64 {
        trunc(a)
    }
}
