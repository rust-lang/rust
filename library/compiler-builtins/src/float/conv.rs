use core::ops::Neg;

use crate::int::{CastFrom, CastInto, Int, MinInt};

use super::Float;

/// Conversions from integers to floats.
///
/// These are hand-optimized bit twiddling code,
/// which unfortunately isn't the easiest kind of code to read.
///
/// The algorithm is explained here: <https://blog.m-ou.se/floats/>
mod int_to_float {
    pub fn u32_to_f32_bits(i: u32) -> u32 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        let a = (i << n) >> 8; // Significant bits, with bit 24 still in tact.
        let b = (i << n) << 24; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 31 & !a)) >> 31); // Add one when we need to round up. Break ties to even.
        let e = 157 - n; // Exponent plus 127, minus one.
        (e << 23) + m // + not |, so the mantissa can overflow into the exponent.
    }

    pub fn u32_to_f64_bits(i: u32) -> u64 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        let m = (i as u64) << (21 + n); // Significant bits, with bit 53 still in tact.
        let e = 1053 - n as u64; // Exponent plus 1023, minus one.
        (e << 52) + m // Bit 53 of m will overflow into e.
    }

    pub fn u64_to_f32_bits(i: u64) -> u32 {
        let n = i.leading_zeros();
        let y = i.wrapping_shl(n);
        let a = (y >> 40) as u32; // Significant bits, with bit 24 still in tact.
        let b = (y >> 8 | y & 0xFFFF) as u32; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 31 & !a)) >> 31); // Add one when we need to round up. Break ties to even.
        let e = if i == 0 { 0 } else { 189 - n }; // Exponent plus 127, minus one, except for zero.
        (e << 23) + m // + not |, so the mantissa can overflow into the exponent.
    }

    pub fn u64_to_f64_bits(i: u64) -> u64 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        let a = (i << n) >> 11; // Significant bits, with bit 53 still in tact.
        let b = (i << n) << 53; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 63 & !a)) >> 63); // Add one when we need to round up. Break ties to even.
        let e = 1085 - n as u64; // Exponent plus 1023, minus one.
        (e << 52) + m // + not |, so the mantissa can overflow into the exponent.
    }

    pub fn u128_to_f32_bits(i: u128) -> u32 {
        let n = i.leading_zeros();
        let y = i.wrapping_shl(n);
        let a = (y >> 104) as u32; // Significant bits, with bit 24 still in tact.
        let b = (y >> 72) as u32 | ((y << 32) >> 32 != 0) as u32; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 31 & !a)) >> 31); // Add one when we need to round up. Break ties to even.
        let e = if i == 0 { 0 } else { 253 - n }; // Exponent plus 127, minus one, except for zero.
        (e << 23) + m // + not |, so the mantissa can overflow into the exponent.
    }

    pub fn u128_to_f64_bits(i: u128) -> u64 {
        let n = i.leading_zeros();
        let y = i.wrapping_shl(n);
        let a = (y >> 75) as u64; // Significant bits, with bit 53 still in tact.
        let b = (y >> 11 | y & 0xFFFF_FFFF) as u64; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 63 & !a)) >> 63); // Add one when we need to round up. Break ties to even.
        let e = if i == 0 { 0 } else { 1149 - n as u64 }; // Exponent plus 1023, minus one, except for zero.
        (e << 52) + m // + not |, so the mantissa can overflow into the exponent.
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
}

// Conversions from signed integers to floats.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_i2f]
    pub extern "C" fn __floatsisf(i: i32) -> f32 {
        let sign_bit = ((i >> 31) as u32) << 31;
        f32::from_bits(int_to_float::u32_to_f32_bits(i.unsigned_abs()) | sign_bit)
    }

    #[arm_aeabi_alias = __aeabi_i2d]
    pub extern "C" fn __floatsidf(i: i32) -> f64 {
        let sign_bit = ((i >> 31) as u64) << 63;
        f64::from_bits(int_to_float::u32_to_f64_bits(i.unsigned_abs()) | sign_bit)
    }

    #[arm_aeabi_alias = __aeabi_l2f]
    pub extern "C" fn __floatdisf(i: i64) -> f32 {
        let sign_bit = ((i >> 63) as u32) << 31;
        f32::from_bits(int_to_float::u64_to_f32_bits(i.unsigned_abs()) | sign_bit)
    }

    #[arm_aeabi_alias = __aeabi_l2d]
    pub extern "C" fn __floatdidf(i: i64) -> f64 {
        let sign_bit = ((i >> 63) as u64) << 63;
        f64::from_bits(int_to_float::u64_to_f64_bits(i.unsigned_abs()) | sign_bit)
    }

    #[cfg_attr(target_os = "uefi", unadjusted_on_win64)]
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        let sign_bit = ((i >> 127) as u32) << 31;
        f32::from_bits(int_to_float::u128_to_f32_bits(i.unsigned_abs()) | sign_bit)
    }

    #[cfg_attr(target_os = "uefi", unadjusted_on_win64)]
    pub extern "C" fn __floattidf(i: i128) -> f64 {
        let sign_bit = ((i >> 127) as u64) << 63;
        f64::from_bits(int_to_float::u128_to_f64_bits(i.unsigned_abs()) | sign_bit)
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
/// - `map_inbounds`: apply this transformation to integers that are within range (add the sign
///    back).
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
    let int_max_exp = F::EXPONENT_BIAS + I::MAX.ilog2() + 1;
    let foobar = F::EXPONENT_BIAS + I::UnsignedInt::BITS - 1;

    if fbits < F::ONE.to_bits() {
        // < 0 gets rounded to 0
        I::ZERO
    } else if fbits < F::Int::cast_from(int_max_exp) << F::SIGNIFICAND_BITS {
        // >= 1, < integer max
        let m_base = if I::UnsignedInt::BITS >= F::Int::BITS {
            I::UnsignedInt::cast_from(fbits) << (I::BITS - F::SIGNIFICAND_BITS - 1)
        } else {
            I::UnsignedInt::cast_from(fbits >> (F::SIGNIFICAND_BITS - I::BITS + 1))
        };

        // Set the implicit 1-bit.
        let m: I::UnsignedInt = I::UnsignedInt::ONE << (I::BITS - 1) | m_base;

        // Shift based on the exponent and bias.
        let s: u32 = (foobar) - u32::cast_from(fbits >> F::SIGNIFICAND_BITS);

        let unsigned = m >> s;
        map_inbounds(I::from_unsigned(unsigned))
    } else if fbits <= F::EXPONENT_MASK {
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

    #[win64_128bit_abi_hack]
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

    #[win64_128bit_abi_hack]
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

    #[win64_128bit_abi_hack]
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

    #[win64_128bit_abi_hack]
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
