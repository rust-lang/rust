/// Conversions from integers to floats.
///
/// These are hand-optimized bit twiddling code,
/// which unfortunately isn't the easiest kind of code to read.
///
/// The algorithm is explained here: https://blog.m-ou.se/floats/
mod int_to_float {
    pub fn u32_to_f32_bits(i: u32) -> u32 {
        if i == 0 {
            return 0;
        }
        let n = i.leading_zeros();
        let a = i << n >> 8; // Significant bits, with bit 24 still in tact.
        let b = i << n << 24; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 31 & !a)) >> 31); // Add one when we need to round up. Break ties to even.
        let e = 157 - n as u32; // Exponent plus 127, minus one.
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
        let a = (i << n >> 11) as u64; // Significant bits, with bit 53 still in tact.
        let b = (i << n << 53) as u64; // Insignificant bits, only relevant for rounding.
        let m = a + ((b - (b >> 63 & !a)) >> 63); // Add one when we need to round up. Break ties to even.
        let e = 1085 - n as u64; // Exponent plus 1023, minus one.
        (e << 52) + m // + not |, so the mantissa can overflow into the exponent.
    }

    pub fn u128_to_f32_bits(i: u128) -> u32 {
        let n = i.leading_zeros();
        let y = i.wrapping_shl(n);
        let a = (y >> 104) as u32; // Significant bits, with bit 24 still in tact.
        let b = (y >> 72) as u32 | (y << 32 >> 32 != 0) as u32; // Insignificant bits, only relevant for rounding.
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

    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __floatuntisf(i: u128) -> f32 {
        f32::from_bits(int_to_float::u128_to_f32_bits(i))
    }

    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
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

    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __floattisf(i: i128) -> f32 {
        let sign_bit = ((i >> 127) as u32) << 31;
        f32::from_bits(int_to_float::u128_to_f32_bits(i.unsigned_abs()) | sign_bit)
    }

    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __floattidf(i: i128) -> f64 {
        let sign_bit = ((i >> 127) as u64) << 63;
        f64::from_bits(int_to_float::u128_to_f64_bits(i.unsigned_abs()) | sign_bit)
    }
}

// Conversions from floats to unsigned integers.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_f2uiz]
    pub extern "C" fn __fixunssfsi(f: f32) -> u32 {
        let fbits = f.to_bits();
        if fbits < 127 << 23 { // >= 0, < 1
            0
        } else if fbits < 159 << 23 { // >= 1, < max
            let m = 1 << 31 | fbits << 8; // Mantissa and the implicit 1-bit.
            let s = 158 - (fbits >> 23); // Shift based on the exponent and bias.
            m >> s
        } else if fbits <= 255 << 23 { // >= max (incl. inf)
            u32::MAX
        } else { // Negative or NaN
            0
        }
    }

    #[arm_aeabi_alias = __aeabi_f2ulz]
    pub extern "C" fn __fixunssfdi(f: f32) -> u64 {
        let fbits = f.to_bits();
        if fbits < 127 << 23 { // >= 0, < 1
            0
        } else if fbits < 191 << 23 { // >= 1, < max
            let m = 1 << 63 | (fbits as u64) << 40; // Mantissa and the implicit 1-bit.
            let s = 190 - (fbits >> 23); // Shift based on the exponent and bias.
            m >> s
        } else if fbits <= 255 << 23 { // >= max (incl. inf)
            u64::MAX
        } else { // Negative or NaN
            0
        }
    }

    #[cfg_attr(target_feature = "llvm14-builtins-abi", win64_128bit_abi_hack)]
    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __fixunssfti(f: f32) -> u128 {
        let fbits = f.to_bits();
        if fbits < 127 << 23 { // >= 0, < 1
            0
        } else if fbits < 255 << 23 { // >= 1, < inf
            let m = 1 << 127 | (fbits as u128) << 104; // Mantissa and the implicit 1-bit.
            let s = 254 - (fbits >> 23); // Shift based on the exponent and bias.
            m >> s
        } else if fbits == 255 << 23 { // == inf
            u128::MAX
        } else { // Negative or NaN
            0
        }
    }

    #[arm_aeabi_alias = __aeabi_d2uiz]
    pub extern "C" fn __fixunsdfsi(f: f64) -> u32 {
        let fbits = f.to_bits();
        if fbits < 1023 << 52 { // >= 0, < 1
            0
        } else if fbits < 1055 << 52 { // >= 1, < max
            let m = 1 << 31 | (fbits >> 21) as u32; // Mantissa and the implicit 1-bit.
            let s = 1054 - (fbits >> 52); // Shift based on the exponent and bias.
            m >> s
        } else if fbits <= 2047 << 52 { // >= max (incl. inf)
            u32::MAX
        } else { // Negative or NaN
            0
        }
    }

    #[arm_aeabi_alias = __aeabi_d2ulz]
    pub extern "C" fn __fixunsdfdi(f: f64) -> u64 {
        let fbits = f.to_bits();
        if fbits < 1023 << 52 { // >= 0, < 1
            0
        } else if fbits < 1087 << 52 { // >= 1, < max
            let m = 1 << 63 | fbits << 11; // Mantissa and the implicit 1-bit.
            let s = 1086 - (fbits >> 52); // Shift based on the exponent and bias.
            m >> s
        } else if fbits <= 2047 << 52 { // >= max (incl. inf)
            u64::MAX
        } else { // Negative or NaN
            0
        }
    }

    #[cfg_attr(target_feature = "llvm14-builtins-abi", win64_128bit_abi_hack)]
    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __fixunsdfti(f: f64) -> u128 {
        let fbits = f.to_bits();
        if fbits < 1023 << 52 { // >= 0, < 1
            0
        } else if fbits < 1151 << 52 { // >= 1, < max
            let m = 1 << 127 | (fbits as u128) << 75; // Mantissa and the implicit 1-bit.
            let s = 1150 - (fbits >> 52); // Shift based on the exponent and bias.
            m >> s
        } else if fbits <= 2047 << 52 { // >= max (incl. inf)
            u128::MAX
        } else { // Negative or NaN
            0
        }
    }
}

// Conversions from floats to signed integers.
intrinsics! {
    #[arm_aeabi_alias = __aeabi_f2iz]
    pub extern "C" fn __fixsfsi(f: f32) -> i32 {
        let fbits = f.to_bits() & !0 >> 1; // Remove sign bit.
        if fbits < 127 << 23 { // >= 0, < 1
            0
        } else if fbits < 158 << 23 { // >= 1, < max
            let m = 1 << 31 | fbits << 8; // Mantissa and the implicit 1-bit.
            let s = 158 - (fbits >> 23); // Shift based on the exponent and bias.
            let u = (m >> s) as i32; // Unsigned result.
            if f.is_sign_negative() { -u } else { u }
        } else if fbits <= 255 << 23 { // >= max (incl. inf)
            if f.is_sign_negative() { i32::MIN } else { i32::MAX }
        } else { // NaN
            0
        }
    }

    #[arm_aeabi_alias = __aeabi_f2lz]
    pub extern "C" fn __fixsfdi(f: f32) -> i64 {
        let fbits = f.to_bits() & !0 >> 1; // Remove sign bit.
        if fbits < 127 << 23 { // >= 0, < 1
            0
        } else if fbits < 190 << 23 { // >= 1, < max
            let m = 1 << 63 | (fbits as u64) << 40; // Mantissa and the implicit 1-bit.
            let s = 190 - (fbits >> 23); // Shift based on the exponent and bias.
            let u = (m >> s) as i64; // Unsigned result.
            if f.is_sign_negative() { -u } else { u }
        } else if fbits <= 255 << 23 { // >= max (incl. inf)
            if f.is_sign_negative() { i64::MIN } else { i64::MAX }
        } else { // NaN
            0
        }
    }

    #[cfg_attr(target_feature = "llvm14-builtins-abi", win64_128bit_abi_hack)]
    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __fixsfti(f: f32) -> i128 {
        let fbits = f.to_bits() & !0 >> 1; // Remove sign bit.
        if fbits < 127 << 23 { // >= 0, < 1
            0
        } else if fbits < 254 << 23 { // >= 1, < max
            let m = 1 << 127 | (fbits as u128) << 104; // Mantissa and the implicit 1-bit.
            let s = 254 - (fbits >> 23); // Shift based on the exponent and bias.
            let u = (m >> s) as i128; // Unsigned result.
            if f.is_sign_negative() { -u } else { u }
        } else if fbits <= 255 << 23 { // >= max (incl. inf)
            if f.is_sign_negative() { i128::MIN } else { i128::MAX }
        } else { // NaN
            0
        }
    }

    #[arm_aeabi_alias = __aeabi_d2iz]
    pub extern "C" fn __fixdfsi(f: f64) -> i32 {
        let fbits = f.to_bits() & !0 >> 1; // Remove sign bit.
        if fbits < 1023 << 52 { // >= 0, < 1
            0
        } else if fbits < 1054 << 52 { // >= 1, < max
            let m = 1 << 31 | (fbits >> 21) as u32; // Mantissa and the implicit 1-bit.
            let s = 1054 - (fbits >> 52); // Shift based on the exponent and bias.
            let u = (m >> s) as i32; // Unsigned result.
            if f.is_sign_negative() { -u } else { u }
        } else if fbits <= 2047 << 52 { // >= max (incl. inf)
            if f.is_sign_negative() { i32::MIN } else { i32::MAX }
        } else { // NaN
            0
        }
    }

    #[arm_aeabi_alias = __aeabi_d2lz]
    pub extern "C" fn __fixdfdi(f: f64) -> i64 {
        let fbits = f.to_bits() & !0 >> 1; // Remove sign bit.
        if fbits < 1023 << 52 { // >= 0, < 1
            0
        } else if fbits < 1086 << 52 { // >= 1, < max
            let m = 1 << 63 | fbits << 11; // Mantissa and the implicit 1-bit.
            let s = 1086 - (fbits >> 52); // Shift based on the exponent and bias.
            let u = (m >> s) as i64; // Unsigned result.
            if f.is_sign_negative() { -u } else { u }
        } else if fbits <= 2047 << 52 { // >= max (incl. inf)
            if f.is_sign_negative() { i64::MIN } else { i64::MAX }
        } else { // NaN
            0
        }
    }

    #[cfg_attr(target_feature = "llvm14-builtins-abi", win64_128bit_abi_hack)]
    #[cfg_attr(not(target_feature = "llvm14-builtins-abi"), unadjusted_on_win64)]
    pub extern "C" fn __fixdfti(f: f64) -> i128 {
        let fbits = f.to_bits() & !0 >> 1; // Remove sign bit.
        if fbits < 1023 << 52 { // >= 0, < 1
            0
        } else if fbits < 1150 << 52 { // >= 1, < max
            let m = 1 << 127 | (fbits as u128) << 75; // Mantissa and the implicit 1-bit.
            let s = 1150 - (fbits >> 52); // Shift based on the exponent and bias.
            let u = (m >> s) as i128; // Unsigned result.
            if f.is_sign_negative() { -u } else { u }
        } else if fbits <= 2047 << 52 { // >= max (incl. inf)
            if f.is_sign_negative() { i128::MIN } else { i128::MAX }
        } else { // NaN
            0
        }
    }
}
