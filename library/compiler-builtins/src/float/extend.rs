use float::Float;
use int::{CastInto, Int};

/// Generic conversion from a narrower to a wider IEEE-754 floating-point type
fn extend<F: Float, R: Float>(a: F) -> R
where
    F::Int: CastInto<u64>,
    u64: CastInto<F::Int>,
    u32: CastInto<R::Int>,
    R::Int: CastInto<u32>,
    R::Int: CastInto<u64>,
    u64: CastInto<R::Int>,
    F::Int: CastInto<R::Int>,
{
    let src_zero = F::Int::ZERO;
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
    let dst_min_normal = R::IMPLICIT_BIT;

    let sign_bits_delta = dst_sign_bits - src_sign_bits;
    let exp_bias_delta = dst_exp_bias - src_exp_bias;
    let a_abs = a.repr() & src_abs_mask;
    let mut abs_result = R::Int::ZERO;

    if a_abs.wrapping_sub(src_min_normal) < src_infinity.wrapping_sub(src_min_normal) {
        // a is a normal number.
        // Extend to the destination type by shifting the significand and
        // exponent into the proper position and rebiasing the exponent.
        let abs_dst: R::Int = a_abs.cast();
        let bias_dst: R::Int = exp_bias_delta.cast();
        abs_result = abs_dst.wrapping_shl(sign_bits_delta);
        abs_result += bias_dst.wrapping_shl(dst_sign_bits);
    } else if a_abs >= src_infinity {
        // a is NaN or infinity.
        // Conjure the result by beginning with infinity, then setting the qNaN
        // bit (if needed) and right-aligning the rest of the trailing NaN
        // payload field.
        let qnan_dst: R::Int = (a_abs & src_qnan).cast();
        let nan_code_dst: R::Int = (a_abs & src_nan_code).cast();
        let inf_exp_dst: R::Int = dst_inf_exp.cast();
        abs_result = inf_exp_dst.wrapping_shl(dst_sign_bits);
        abs_result |= qnan_dst.wrapping_shl(sign_bits_delta);
        abs_result |= nan_code_dst.wrapping_shl(sign_bits_delta);
    } else if a_abs != src_zero {
        // a is denormal.
        // Renormalize the significand and clear the leading bit, then insert
        // the correct adjusted exponent in the destination type.
        let scale = a_abs.leading_zeros() - src_min_normal.leading_zeros();
        let abs_dst: R::Int = a_abs.cast();
        let bias_dst: R::Int = (exp_bias_delta - scale + 1).cast();
        abs_result = abs_dst.wrapping_shl(sign_bits_delta + scale);
        abs_result = (abs_result ^ dst_min_normal) | (bias_dst.wrapping_shl(dst_sign_bits));
    }

    let sign_result: R::Int = (a.repr() & src_sign_mask).cast();
    R::from_repr(abs_result | (sign_result.wrapping_shl(dst_bits - src_bits)))
}

intrinsics! {
    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_f2d]
    pub extern "C" fn  __extendsfdf2(a: f32) -> f64 {
        extend(a)
    }

    #[cfg(target_arch = "arm")]
    pub extern "C" fn  __extendsfdf2vfp(a: f32) -> f64 {
        a as f64 // LLVM generate 'fcvtds'
    }
}
