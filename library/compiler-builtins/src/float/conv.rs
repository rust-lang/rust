use float::Float;
use int::{CastInto, Int};

fn int_to_float<I: Int, F: Float>(i: I) -> F
where
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
    let sign = i < I::ZERO;
    let mut x = Int::abs_diff(i, I::ZERO);

    // number of significant digits in the integer
    let i_sd = I::BITS - x.leading_zeros();
    // significant digits for the float, including implicit bit
    let f_sd = F::SIGNIFICAND_BITS + 1;

    // exponent
    let mut exp = i_sd - 1;

    if I::BITS < f_sd {
        return F::from_parts(
            sign,
            (exp + F::EXPONENT_BIAS).cast(),
            x.cast() << (f_sd - exp - 1),
        );
    }

    x = if i_sd > f_sd {
        // start:  0000000000000000000001xxxxxxxxxxxxxxxxxxxxxxPQxxxxxxxxxxxxxxxxxx
        // finish: 000000000000000000000000000000000000001xxxxxxxxxxxxxxxxxxxxxxPQR
        //                                               12345678901234567890123456
        // 1 = the implicit bit
        // P = bit f_sd-1 bits to the right of 1
        // Q = bit f_sd bits to the right of 1
        // R = "or" of all bits to the right of Q
        let f_sd_add2 = f_sd + 2;
        x = if i_sd == (f_sd + 1) {
            x << 1
        } else if i_sd == f_sd_add2 {
            x
        } else {
            (x >> (i_sd - f_sd_add2))
                | Int::from_bool(
                    (x & I::UnsignedInt::MAX).wrapping_shl((I::BITS + f_sd_add2) - i_sd)
                        != Int::ZERO,
                )
        };

        // R |= P
        x |= Int::from_bool((x & four) != I::UnsignedInt::ZERO);
        // round - this step may add a significant bit
        x += Int::ONE;
        // dump Q and R
        x >>= 2;

        // a is now rounded to f_sd or f_sd+1 bits
        if (x & (I::UnsignedInt::ONE << f_sd)) != Int::ZERO {
            x >>= 1;
            exp += 1;
        }
        x
    } else {
        x.wrapping_shl(f_sd - i_sd)
    };

    F::from_parts(sign, (exp + F::EXPONENT_BIAS).cast(), x.cast())
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

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_l2f]
    pub extern "C" fn __floatdisf(i: i64) -> f32 {
        // On x86_64 LLVM will use native instructions for this conversion, we
        // can just do it directly
        if cfg!(target_arch = "x86_64") {
            i as f32
        } else {
            int_to_float(i)
        }
    }

    #[maybe_use_optimized_c_shim]
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

    #[arm_aeabi_alias = __aeabi_ui2f]
    pub extern "C" fn __floatunsisf(i: u32) -> f32 {
        int_to_float(i)
    }

    #[arm_aeabi_alias = __aeabi_ui2d]
    pub extern "C" fn __floatunsidf(i: u32) -> f64 {
        int_to_float(i)
    }

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_ul2f]
    pub extern "C" fn __floatundisf(i: u64) -> f32 {
        int_to_float(i)
    }

    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_ul2d]
    pub extern "C" fn __floatundidf(i: u64) -> f64 {
        int_to_float(i)
    }
}

fn float_to_int<F: Float, I: Int>(f: F) -> I
where
    F::ExpInt: CastInto<u32>,
    u32: CastInto<F::ExpInt>,
    F::Int: CastInto<I>,
{
    // converting NaNs is UB, so we don't consider them

    let sign = f.sign();
    let mut exp = f.exp();

    // if less than one or unsigned & negative
    if (exp < F::EXPONENT_BIAS.cast()) || (!I::SIGNED && sign) {
        return I::ZERO;
    }
    exp -= F::EXPONENT_BIAS.cast();

    // If the value is too large for `I`, saturate.
    let bits: F::ExpInt = I::BITS.cast();
    let max = if I::SIGNED {
        bits - F::ExpInt::ONE
    } else {
        bits
    };
    if max <= exp {
        return if sign {
            // It happens that I::MIN is handled correctly
            I::MIN
        } else {
            I::MAX
        };
    };

    // `0 <= exp < max`

    // If 0 <= exponent < F::SIGNIFICAND_BITS, right shift to get the result. Otherwise, shift left.
    let sig_bits: F::ExpInt = F::SIGNIFICAND_BITS.cast();
    // The larger integer has to be casted into, or else the shift overflows
    let r: I = if F::Int::BITS < I::BITS {
        let tmp: I = if exp < sig_bits {
            f.imp_frac().cast() >> (sig_bits - exp).cast()
        } else {
            f.imp_frac().cast() << (exp - sig_bits).cast()
        };
        tmp
    } else {
        let tmp: F::Int = if exp < sig_bits {
            f.imp_frac() >> (sig_bits - exp).cast()
        } else {
            f.imp_frac() << (exp - sig_bits).cast()
        };
        tmp.cast()
    };

    if sign {
        r.wrapping_neg()
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

    #[arm_aeabi_alias = __aeabi_d2iz]
    pub extern "C" fn __fixdfsi(f: f64) -> i32 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2lz]
    pub extern "C" fn __fixdfdi(f: f64) -> i64 {
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

    #[arm_aeabi_alias = __aeabi_d2uiz]
    pub extern "C" fn __fixunsdfsi(f: f64) -> u32 {
        float_to_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2ulz]
    pub extern "C" fn __fixunsdfdi(f: f64) -> u64 {
        float_to_int(f)
    }
}

// The ABI for the following intrinsics changed in LLVM 14. On Win64, they now
// use Win64 ABI rather than unadjusted ABI. Pick the correct ABI based on the
// llvm14-builtins-abi target feature.

#[cfg(target_feature = "llvm14-builtins-abi")]
intrinsics! {
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        int_to_float(i)
    }

    pub extern "C" fn __floattidf(i: i128) -> f64 {
        int_to_float(i)
    }

    pub extern "C" fn __floatuntisf(i: u128) -> f32 {
        int_to_float(i)
    }

    pub extern "C" fn __floatuntidf(i: u128) -> f64 {
        int_to_float(i)
    }

    #[win64_128bit_abi_hack]
    pub extern "C" fn __fixsfti(f: f32) -> i128 {
        float_to_int(f)
    }

    #[win64_128bit_abi_hack]
    pub extern "C" fn __fixdfti(f: f64) -> i128 {
        float_to_int(f)
    }

    #[win64_128bit_abi_hack]
    pub extern "C" fn __fixunssfti(f: f32) -> u128 {
        float_to_int(f)
    }

    #[win64_128bit_abi_hack]
    pub extern "C" fn __fixunsdfti(f: f64) -> u128 {
        float_to_int(f)
    }
}

#[cfg(not(target_feature = "llvm14-builtins-abi"))]
intrinsics! {
    #[unadjusted_on_win64]
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        int_to_float(i)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __floattidf(i: i128) -> f64 {
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

    #[unadjusted_on_win64]
    pub extern "C" fn __fixsfti(f: f32) -> i128 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixdfti(f: f64) -> i128 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixunssfti(f: f32) -> u128 {
        float_to_int(f)
    }

    #[unadjusted_on_win64]
    pub extern "C" fn __fixunsdfti(f: f64) -> u128 {
        float_to_int(f)
    }
}
