use core::ops::Neg;

use super::Float;
use crate::int::{CastFrom, CastInto, Int, MinInt};

/// Conversions from integers to floats.
///
/// The algorithm is explained here: <https://blog.m-ou.se/floats/>. It roughly does the following:
/// - Calculate a base mantissa by shifting the integer into mantissa position. This gives us a
///   mantissa _with the implicit bit set_!
/// - Figure out if rounding needs to occur by classifying the bits that are to be truncated. Some
///   patterns are used to simplify this. Adjust the mantissa with the result if needed.
/// - Calculate the exponent based on the base-2 logarithm of `i` (leading zeros). Subtract one.
/// - Shift the exponent and add the mantissa to create the final representation. Subtracting one
///   from the exponent (above) accounts for the explicit bit being set in the mantissa.
///
/// # Terminology
///
/// - `i`: the original integer
/// - `i_m`: the integer, shifted fully left (no leading zeros)
/// - `n`: number of leading zeroes
/// - `e`: the resulting exponent. Usually 1 is subtracted to offset the mantissa implicit bit.
/// - `m_base`: the mantissa before adjusting for truncated bits. Implicit bit is usually set.
/// - `adj`: the bits that will be truncated, possibly compressed in some way.
/// - `m`: the resulting mantissa. Implicit bit is usually set.
mod int_to_float {
    use super::*;

    /// Calculate the exponent from the number of leading zeros.
    ///
    /// Usually 1 is subtracted from this function's result, so that a mantissa with the implicit
    /// bit set can be added back later.
    fn exp<I: Int, F: Float<Int: CastFrom<u32>>>(n: u32) -> F::Int {
        F::Int::cast_from(F::EXP_BIAS - 1 + I::BITS - n)
    }

    /// Adjust a mantissa with dropped bits to perform correct rounding.
    ///
    /// The dropped bits should be exactly the bits that get truncated (left-aligned), but they
    /// can be combined or compressed in some way that simplifies operations.
    fn m_adj<F: Float>(m_base: F::Int, dropped_bits: F::Int) -> F::Int {
        // Branchlessly extract a `1` if rounding up should happen, 0 otherwise
        // This accounts for rounding to even.
        let adj = (dropped_bits - ((dropped_bits >> (F::BITS - 1)) & !m_base)) >> (F::BITS - 1);

        // Add one when we need to round up. Break ties to even.
        m_base + adj
    }

    /// Shift the exponent to its position and add the mantissa.
    ///
    /// If the mantissa has the implicit bit set, the exponent should be one less than its actual
    /// value to cancel it out.
    fn repr<F: Float>(e: F::Int, m: F::Int) -> F::Int {
        // + rather than | so the mantissa can overflow into the exponent
        (e << F::SIG_BITS) + m
    }

    /// Shift distance from a left-aligned integer to a smaller float.
    fn shift_f_lt_i<I: Int, F: Float>() -> u32 {
        (I::BITS - F::BITS) + F::EXP_BITS
    }

    /// Shift distance from an integer with `n` leading zeros to a smaller float.
    fn shift_f_gt_i<I: Int, F: Float>(n: u32) -> u32 {
        F::SIG_BITS - I::BITS + 1 + n
    }

    /// Perform a signed operation as unsigned, then add the sign back.
    pub fn signed<I, F, Conv>(i: I, conv: Conv) -> F
    where
        F: Float,
        I: Int,
        F::Int: CastFrom<I>,
        Conv: Fn(I::UnsignedInt) -> F::Int,
    {
        let sign_bit = F::Int::cast_from(i >> (I::BITS - 1)) << (F::BITS - 1);
        F::from_bits(conv(i.unsigned_abs()) | sign_bit)
    }

    pub fn u32_to_f32_bits(i: u32) -> u32 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        // Mantissa with implicit bit set (significant bits)
        let m_base = (i << n) >> f32::EXP_BITS;
        // Bits that will be dropped (insignificant bits)
        let adj = (i << n) << (f32::SIG_BITS + 1);
        let m = m_adj::<f32>(m_base, adj);
        let e = exp::<u32, f32>(n) - 1;
        repr::<f32>(e, m)
    }

    pub fn u32_to_f64_bits(i: u32) -> u64 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        // Mantissa with implicit bit set
        let m = (i as u64) << shift_f_gt_i::<u32, f64>(n);
        let e = exp::<u32, f64>(n) - 1;
        repr::<f64>(e, m)
    }

    #[cfg(f128_enabled)]
    pub fn u32_to_f128_bits(i: u32) -> u128 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();

        // Shift into mantissa position that is correct for the type, but shifted into the lower
        // 64 bits over so can can avoid 128-bit math.
        let m = (i as u64) << (shift_f_gt_i::<u32, f128>(n) - 64);
        let e = exp::<u32, f128>(n) as u64 - 1;
        // High 64 bits of f128 representation.
        let h = (e << (f128::SIG_BITS - 64)) + m;

        // Shift back to the high bits, the rest of the mantissa will always be 0.
        (h as u128) << 64
    }

    pub fn u64_to_f32_bits(i: u64) -> u32 {
        let n = i.leading_zeros();
        let i_m = i.wrapping_shl(n);
        // Mantissa with implicit bit set
        let m_base: u32 = (i_m >> shift_f_lt_i::<u64, f32>()) as u32;
        // The entire lower half of `i` will be truncated (masked portion), plus the
        // next `EXP_BITS` bits.
        let adj = ((i_m >> f32::EXP_BITS) | i_m & 0xFFFF) as u32;
        let m = m_adj::<f32>(m_base, adj);
        let e = if i == 0 { 0 } else { exp::<u64, f32>(n) - 1 };
        repr::<f32>(e, m)
    }

    pub fn u64_to_f64_bits(i: u64) -> u64 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        // Mantissa with implicit bit set
        let m_base = (i << n) >> f64::EXP_BITS;
        let adj = (i << n) << (f64::SIG_BITS + 1);
        let m = m_adj::<f64>(m_base, adj);
        let e = exp::<u64, f64>(n) - 1;
        repr::<f64>(e, m)
    }

    #[cfg(f128_enabled)]
    pub fn u64_to_f128_bits(i: u64) -> u128 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        // Mantissa with implicit bit set
        let m = (i as u128) << shift_f_gt_i::<u64, f128>(n);
        let e = exp::<u64, f128>(n) - 1;
        repr::<f128>(e, m)
    }

    pub fn u128_to_f32_bits(i: u128) -> u32 {
        let n = i.leading_zeros();
        let i_m = i.wrapping_shl(n); // Mantissa, shifted so the first bit is nonzero
        let m_base: u32 = (i_m >> shift_f_lt_i::<u128, f32>()) as u32;

        // Within the upper `F::BITS`, everything except for the signifcand
        // gets truncated
        let d1: u32 = (i_m >> (u128::BITS - f32::BITS - f32::SIG_BITS - 1)).cast();

        // The entire rest of `i_m` gets truncated. Zero the upper `F::BITS` then just
        // check if it is nonzero.
        let d2: u32 = (i_m << f32::BITS >> f32::BITS != 0).into();
        let adj = d1 | d2;

        // Mantissa with implicit bit set
        let m = m_adj::<f32>(m_base, adj);
        let e = if i == 0 { 0 } else { exp::<u128, f32>(n) - 1 };
        repr::<f32>(e, m)
    }

    pub fn u128_to_f64_bits(i: u128) -> u64 {
        let n = i.leading_zeros();
        let i_m = i.wrapping_shl(n);
        // Mantissa with implicit bit set
        let m_base: u64 = (i_m >> shift_f_lt_i::<u128, f64>()) as u64;
        // The entire lower half of `i` will be truncated (masked portion), plus the
        // next `EXP_BITS` bits.
        let adj = ((i_m >> f64::EXP_BITS) | i_m & 0xFFFF_FFFF) as u64;
        let m = m_adj::<f64>(m_base, adj);
        let e = if i == 0 { 0 } else { exp::<u128, f64>(n) - 1 };
        repr::<f64>(e, m)
    }

    #[cfg(f128_enabled)]
    pub fn u128_to_f128_bits(i: u128) -> u128 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        // Mantissa with implicit bit set
        let m_base = (i << n) >> f128::EXP_BITS;
        let adj = (i << n) << (f128::SIG_BITS + 1);
        let m = m_adj::<f128>(m_base, adj);
        let e = exp::<u128, f128>(n) - 1;
        repr::<f128>(e, m)
    }
}

// Conversions from unsigned integers to floats.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_ui2f]
    pub extern "C" fn __floatunsisf(i: u32) -> f32 {
        f32::from_bits(int_to_float::u32_to_f32_bits(i))
    }

    #[arm_aeabi_alias = __aeabi_ui2d]
    pub extern "C" fn __floatunsidf(i: u32) -> f64 {
        f64::from_bits(int_to_float::u32_to_f64_bits(i))
    }

    #[arm_aeabi_alias = __aeabi_ul2f]
    pub extern "C" fn __floatundisf(i: u64) -> f32 {
        f32::from_bits(int_to_float::u64_to_f32_bits(i))
    }

    #[arm_aeabi_alias = __aeabi_ul2d]
    pub extern "C" fn __floatundidf(i: u64) -> f64 {
        f64::from_bits(int_to_float::u64_to_f64_bits(i))
    }

    #[cfg_attr(target_os = "uefi", unadjusted_on_win64)]
    pub extern "C" fn __floatuntisf(i: u128) -> f32 {
        f32::from_bits(int_to_float::u128_to_f32_bits(i))
    }

    #[cfg_attr(target_os = "uefi", unadjusted_on_win64)]
    pub extern "C" fn __floatuntidf(i: u128) -> f64 {
        f64::from_bits(int_to_float::u128_to_f64_bits(i))
    }

    #[ppc_alias = __floatunsikf]
    #[cfg(f128_enabled)]
    pub extern "C" fn __floatunsitf(i: u32) -> f128 {
        f128::from_bits(int_to_float::u32_to_f128_bits(i))
    }

    #[ppc_alias = __floatundikf]
    #[cfg(f128_enabled)]
    pub extern "C" fn __floatunditf(i: u64) -> f128 {
        f128::from_bits(int_to_float::u64_to_f128_bits(i))
    }

    #[ppc_alias = __floatuntikf]
    #[cfg(f128_enabled)]
    pub extern "C" fn __floatuntitf(i: u128) -> f128 {
        f128::from_bits(int_to_float::u128_to_f128_bits(i))
    }
}

// Conversions from signed integers to floats.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_i2f]
    pub extern "C" fn __floatsisf(i: i32) -> f32 {
        int_to_float::signed(i, int_to_float::u32_to_f32_bits)
    }

    #[arm_aeabi_alias = __aeabi_i2d]
    pub extern "C" fn __floatsidf(i: i32) -> f64 {
        int_to_float::signed(i, int_to_float::u32_to_f64_bits)
    }

    #[arm_aeabi_alias = __aeabi_l2f]
    pub extern "C" fn __floatdisf(i: i64) -> f32 {
        int_to_float::signed(i, int_to_float::u64_to_f32_bits)
    }

    #[arm_aeabi_alias = __aeabi_l2d]
    pub extern "C" fn __floatdidf(i: i64) -> f64 {
        int_to_float::signed(i, int_to_float::u64_to_f64_bits)
    }

    #[cfg_attr(target_os = "uefi", unadjusted_on_win64)]
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        int_to_float::signed(i, int_to_float::u128_to_f32_bits)
    }

    #[cfg_attr(target_os = "uefi", unadjusted_on_win64)]
    pub extern "C" fn __floattidf(i: i128) -> f64 {
        int_to_float::signed(i, int_to_float::u128_to_f64_bits)
    }

    #[ppc_alias = __floatsikf]
    #[cfg(f128_enabled)]
    pub extern "C" fn __floatsitf(i: i32) -> f128 {
        int_to_float::signed(i, int_to_float::u32_to_f128_bits)
    }

    #[ppc_alias = __floatdikf]
    #[cfg(f128_enabled)]
    pub extern "C" fn __floatditf(i: i64) -> f128 {
        int_to_float::signed(i, int_to_float::u64_to_f128_bits)
    }

    #[ppc_alias = __floattikf]
    #[cfg(f128_enabled)]
    pub extern "C" fn __floattitf(i: i128) -> f128 {
        int_to_float::signed(i, int_to_float::u128_to_f128_bits)
    }
}

/// Generic float to unsigned int conversions.
fn float_to_unsigned_int<F, U>(f: F) -> U
where
    F: Float,
    U: Int<UnsignedInt = U>,
    F::Int: CastInto<U>,
    F::Int: CastFrom<u32>,
    F::Int: CastInto<U::UnsignedInt>,
    u32: CastFrom<F::Int>,
{
    float_to_int_inner::<F, U, _, _>(f.to_bits(), |i: U| i, || U::MAX)
}

/// Generic float to signed int conversions.
fn float_to_signed_int<F, I>(f: F) -> I
where
    F: Float,
    I: Int + Neg<Output = I>,
    I::UnsignedInt: Int,
    F::Int: CastInto<I::UnsignedInt>,
    F::Int: CastFrom<u32>,
    u32: CastFrom<F::Int>,
{
    float_to_int_inner::<F, I, _, _>(
        f.to_bits() & !F::SIGN_MASK,
        |i: I| if f.is_sign_negative() { -i } else { i },
        || if f.is_sign_negative() { I::MIN } else { I::MAX },
    )
}

/// Float to int conversions, generic for both signed and unsigned.
///
/// Parameters:
/// - `fbits`: `abg(f)` bitcasted to an integer.
/// - `map_inbounds`: apply this transformation to integers that are within range (add the sign back).
/// - `out_of_bounds`: return value when out of range for `I`.
fn float_to_int_inner<F, I, FnFoo, FnOob>(
    fbits: F::Int,
    map_inbounds: FnFoo,
    out_of_bounds: FnOob,
) -> I
where
    F: Float,
    I: Int,
    FnFoo: FnOnce(I) -> I,
    FnOob: FnOnce() -> I,
    I::UnsignedInt: Int,
    F::Int: CastInto<I::UnsignedInt>,
    F::Int: CastFrom<u32>,
    u32: CastFrom<F::Int>,
{
    let int_max_exp = F::EXP_BIAS + I::MAX.ilog2() + 1;
    let foobar = F::EXP_BIAS + I::UnsignedInt::BITS - 1;

    if fbits < F::ONE.to_bits() {
        // < 0 gets rounded to 0
        I::ZERO
    } else if fbits < F::Int::cast_from(int_max_exp) << F::SIG_BITS {
        // >= 1, < integer max
        let m_base = if I::UnsignedInt::BITS >= F::Int::BITS {
            I::UnsignedInt::cast_from(fbits) << (I::BITS - F::SIG_BITS - 1)
        } else {
            I::UnsignedInt::cast_from(fbits >> (F::SIG_BITS - I::BITS + 1))
        };

        // Set the implicit 1-bit.
        let m: I::UnsignedInt = (I::UnsignedInt::ONE << (I::BITS - 1)) | m_base;

        // Shift based on the exponent and bias.
        let s: u32 = (foobar) - u32::cast_from(fbits >> F::SIG_BITS);

        let unsigned = m >> s;
        map_inbounds(I::from_unsigned(unsigned))
    } else if fbits <= F::EXP_MASK {
        // >= max (incl. inf)
        out_of_bounds()
    } else {
        I::ZERO
    }
}

// Conversions from floats to unsigned integers.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_f2uiz]
    pub extern "C" fn __fixunssfsi(f: f32) -> u32 {
        float_to_unsigned_int(f)
    }

    #[arm_aeabi_alias = __aeabi_f2ulz]
    pub extern "C" fn __fixunssfdi(f: f32) -> u64 {
        float_to_unsigned_int(f)
    }

    pub extern "C" fn __fixunssfti(f: f32) -> u128 {
        float_to_unsigned_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2uiz]
    pub extern "C" fn __fixunsdfsi(f: f64) -> u32 {
        float_to_unsigned_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2ulz]
    pub extern "C" fn __fixunsdfdi(f: f64) -> u64 {
        float_to_unsigned_int(f)
    }

    pub extern "C" fn __fixunsdfti(f: f64) -> u128 {
        float_to_unsigned_int(f)
    }

    #[ppc_alias = __fixunskfsi]
    #[cfg(f128_enabled)]
    pub extern "C" fn __fixunstfsi(f: f128) -> u32 {
        float_to_unsigned_int(f)
    }

    #[ppc_alias = __fixunskfdi]
    #[cfg(f128_enabled)]
    pub extern "C" fn __fixunstfdi(f: f128) -> u64 {
        float_to_unsigned_int(f)
    }

    #[ppc_alias = __fixunskfti]
    #[cfg(f128_enabled)]
    pub extern "C" fn __fixunstfti(f: f128) -> u128 {
        float_to_unsigned_int(f)
    }
}

// Conversions from floats to signed integers.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_f2iz]
    pub extern "C" fn __fixsfsi(f: f32) -> i32 {
        float_to_signed_int(f)
    }

    #[arm_aeabi_alias = __aeabi_f2lz]
    pub extern "C" fn __fixsfdi(f: f32) -> i64 {
        float_to_signed_int(f)
    }

    pub extern "C" fn __fixsfti(f: f32) -> i128 {
        float_to_signed_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2iz]
    pub extern "C" fn __fixdfsi(f: f64) -> i32 {
        float_to_signed_int(f)
    }

    #[arm_aeabi_alias = __aeabi_d2lz]
    pub extern "C" fn __fixdfdi(f: f64) -> i64 {
        float_to_signed_int(f)
    }

    pub extern "C" fn __fixdfti(f: f64) -> i128 {
        float_to_signed_int(f)
    }

    #[ppc_alias = __fixkfsi]
    #[cfg(f128_enabled)]
    pub extern "C" fn __fixtfsi(f: f128) -> i32 {
        float_to_signed_int(f)
    }

    #[ppc_alias = __fixkfdi]
    #[cfg(f128_enabled)]
    pub extern "C" fn __fixtfdi(f: f128) -> i64 {
        float_to_signed_int(f)
    }

    #[ppc_alias = __fixkfti]
    #[cfg(f128_enabled)]
    pub extern "C" fn __fixtfti(f: f128) -> i128 {
        float_to_signed_int(f)
    }
}
