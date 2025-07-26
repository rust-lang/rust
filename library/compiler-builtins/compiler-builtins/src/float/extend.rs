use crate::float::Float;
use crate::int::{CastInto, Int, MinInt};

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
    let src_sig_bits = F::SIG_BITS;
    let src_exp_bias = F::EXP_BIAS;
    let src_min_normal = F::IMPLICIT_BIT;
    let src_infinity = F::EXP_MASK;
    let src_sign_mask = F::SIGN_MASK;
    let src_abs_mask = src_sign_mask - src_one;
    let src_qnan = F::SIG_MASK;
    let src_nan_code = src_qnan - src_one;

    let dst_bits = R::BITS;
    let dst_sig_bits = R::SIG_BITS;
    let dst_inf_exp = R::EXP_SAT;
    let dst_exp_bias = R::EXP_BIAS;
    let dst_min_normal = R::IMPLICIT_BIT;

    let sig_bits_delta = dst_sig_bits - src_sig_bits;
    let exp_bias_delta = dst_exp_bias - src_exp_bias;
    let a_abs = a.to_bits() & src_abs_mask;
    let mut abs_result = R::Int::ZERO;

    if a_abs.wrapping_sub(src_min_normal) < src_infinity.wrapping_sub(src_min_normal) {
        // a is a normal number.
        // Extend to the destination type by shifting the significand and
        // exponent into the proper position and rebiasing the exponent.
        let abs_dst: R::Int = a_abs.cast();
        let bias_dst: R::Int = exp_bias_delta.cast();
        abs_result = abs_dst.wrapping_shl(sig_bits_delta);
        abs_result += bias_dst.wrapping_shl(dst_sig_bits);
    } else if a_abs >= src_infinity {
        // a is NaN or infinity.
        // Conjure the result by beginning with infinity, then setting the qNaN
        // bit (if needed) and right-aligning the rest of the trailing NaN
        // payload field.
        let qnan_dst: R::Int = (a_abs & src_qnan).cast();
        let nan_code_dst: R::Int = (a_abs & src_nan_code).cast();
        let inf_exp_dst: R::Int = dst_inf_exp.cast();
        abs_result = inf_exp_dst.wrapping_shl(dst_sig_bits);
        abs_result |= qnan_dst.wrapping_shl(sig_bits_delta);
        abs_result |= nan_code_dst.wrapping_shl(sig_bits_delta);
    } else if a_abs != src_zero {
        // a is denormal.
        // Renormalize the significand and clear the leading bit, then insert
        // the correct adjusted exponent in the destination type.
        let scale = a_abs.leading_zeros() - src_min_normal.leading_zeros();
        let abs_dst: R::Int = a_abs.cast();
        let bias_dst: R::Int = (exp_bias_delta - scale + 1).cast();
        abs_result = abs_dst.wrapping_shl(sig_bits_delta + scale);
        abs_result = (abs_result ^ dst_min_normal) | (bias_dst.wrapping_shl(dst_sig_bits));
    }

    let sign_result: R::Int = (a.to_bits() & src_sign_mask).cast();
    R::from_bits(abs_result | (sign_result.wrapping_shl(dst_bits - src_bits)))
}

intrinsics! {
    #[aapcs_on_arm]
    #[arm_aeabi_alias = __aeabi_f2d]
    pub extern "C" fn  __extendsfdf2(a: f32) -> f64 {
        extend(a)
    }
}

intrinsics! {
    #[aapcs_on_arm]
    #[apple_f16_arg_abi]
    #[arm_aeabi_alias = __aeabi_h2f]
    #[cfg(f16_enabled)]
    pub extern "C" fn __extendhfsf2(a: f16) -> f32 {
        extend(a)
    }

    #[aapcs_on_arm]
    #[apple_f16_arg_abi]
    #[cfg(f16_enabled)]
    pub extern "C" fn __gnu_h2f_ieee(a: f16) -> f32 {
        extend(a)
    }

    #[aapcs_on_arm]
    #[apple_f16_arg_abi]
    #[cfg(f16_enabled)]
    pub extern "C" fn __extendhfdf2(a: f16) -> f64 {
        extend(a)
    }

    #[aapcs_on_arm]
    #[ppc_alias = __extendhfkf2]
    #[cfg(all(f16_enabled, f128_enabled))]
    pub extern "C" fn __extendhftf2(a: f16) -> f128 {
        extend(a)
    }

    #[aapcs_on_arm]
    #[ppc_alias = __extendsfkf2]
    #[cfg(f128_enabled)]
    pub extern "C" fn __extendsftf2(a: f32) -> f128 {
        extend(a)
    }

    #[aapcs_on_arm]
    #[ppc_alias = __extenddfkf2]
    #[cfg(f128_enabled)]
    pub extern "C" fn __extenddftf2(a: f64) -> f128 {
        extend(a)
    }
}
