use float::Float;
use int::{CastInto, Int};

/// Generic conversion from a wider to a narrower IEEE-754 floating-point type
fn truncate<F: Float, R: Float>(a: F) -> R
where
    F::Int: CastInto<u64>,
    u64: CastInto<F::Int>,
    F::Int: CastInto<u32>,
    u32: CastInto<F::Int>,
    u32: CastInto<R::Int>,
    R::Int: CastInto<u32>,
    F::Int: CastInto<R::Int>,
{
    let src_one = F::Int::ONE;
    let src_bits = F::BITS;
    let src_sign_bits = F::SIGNIFICAND_BITS;
    let src_exp_bias = F::EXPONENT_BIAS;
    let src_min_normal = F::IMPLICIT_BIT;
    let src_infinity = F::EXPONENT_MASK;
    let src_sign_mask = F::SIGN_MASK as F::Int;
    let src_abs_mask = src_sign_mask - src_one;
    let src_qnan = F::SIGNIFICAND_MASK;
    let src_nan_code = src_qnan - src_one;

    let dst_bits = R::BITS;
    let dst_sign_bits = R::SIGNIFICAND_BITS;
    let dst_inf_exp = R::EXPONENT_MAX;
    let dst_exp_bias = R::EXPONENT_BIAS;

    let dst_zero = R::Int::ZERO;
    let dst_one = R::Int::ONE;
    let dst_qnan = R::SIGNIFICAND_MASK;
    let dst_nan_code = dst_qnan - dst_one;

    let round_mask = (src_one << src_sign_bits - dst_sign_bits) - src_one;
    let half = src_one << src_sign_bits - dst_sign_bits - 1;
    let underflow_exp = src_exp_bias + 1 - dst_exp_bias;
    let overflow_exp = src_exp_bias + dst_inf_exp - dst_exp_bias;
    let underflow: F::Int = underflow_exp.cast(); // << src_sign_bits;
    let overflow: F::Int = overflow_exp.cast(); //<< src_sign_bits;

    let a_abs = a.repr() & src_abs_mask;
    let sign = a.repr() & src_sign_mask;
    let mut abs_result: R::Int;

    let src_underflow = underflow << src_sign_bits;
    let src_overflow = overflow << src_sign_bits;

    if a_abs.wrapping_sub(src_underflow) < a_abs.wrapping_sub(src_overflow) {
        // The exponent of a is within the range of normal numbers
        let bias_delta: R::Int = (src_exp_bias - dst_exp_bias).cast();
        abs_result = a_abs.cast();
        abs_result = abs_result >> src_sign_bits - dst_sign_bits;
        abs_result = abs_result - bias_delta.wrapping_shl(dst_sign_bits);
        let round_bits: F::Int = a_abs & round_mask;
        abs_result += if round_bits > half {
            dst_one
        } else {
            abs_result & dst_one
        };
    } else if a_abs > src_infinity {
        // a is NaN.
        // Conjure the result by beginning with infinity, setting the qNaN
        // bit and inserting the (truncated) trailing NaN field
        let nan_result: R::Int = (a_abs & src_nan_code).cast();
        abs_result = dst_inf_exp.cast();
        abs_result = abs_result.wrapping_shl(dst_sign_bits);
        abs_result |= dst_qnan;
        abs_result |= (nan_result >> (src_sign_bits - dst_sign_bits)) & dst_nan_code;
    } else if a_abs >= src_overflow {
        // a overflows to infinity.
        abs_result = dst_inf_exp.cast();
        abs_result = abs_result.wrapping_shl(dst_sign_bits);
    } else {
        // a underflows on conversion to the destination type or is an exact
        // zero. The result may be a denormal or zero. Extract the exponent
        // to get the shift amount for the denormalization.
        let a_exp = a_abs >> src_sign_bits;
        let mut shift: u32 = a_exp.cast();
        shift = src_exp_bias - dst_exp_bias - shift + 1;

        let significand = (a.repr() & src_sign_mask) | src_min_normal;
        if shift > src_sign_bits {
            abs_result = dst_zero;
        } else {
            let sticky = significand << src_bits - shift;
            let mut denormalized_significand: R::Int = significand.cast();
            let sticky_shift: u32 = sticky.cast();
            denormalized_significand = denormalized_significand >> (shift | sticky_shift);
            abs_result = denormalized_significand >> src_sign_bits - dst_sign_bits;
            let round_bits = denormalized_significand & round_mask.cast();
            if round_bits > half.cast() {
                abs_result += dst_one; // Round to nearest
            } else if round_bits == half.cast() {
                abs_result += abs_result & dst_one; // Ties to even
            }
        }
    }
    // Finally apply the sign bit
    let s = sign >> src_bits - dst_bits;
    R::from_repr(abs_result | s.cast())
}

intrinsics! {
    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_d2f]
    pub extern "C" fn  __truncdfsf2(a: f64) -> f32 {
        truncate(a)
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn  __truncdfsf2vfp(a: f64) -> f32 {
        a as f32
    }
}
