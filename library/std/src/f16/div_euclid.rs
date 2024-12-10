#[cfg(test)]
mod tests;

/// Software implementation of `f16::div_euclid`.
#[allow(dead_code)] // f16::div_euclid is cfg(not(test))
pub(crate) fn div_euclid(a: f16, b: f16) -> f16 {
    if let Some((a_neg, a_exp, a_m)) = normal_form(a)
        && let Some((b_neg, b_exp, b_m)) = normal_form(b)
    {
        let exp = a_exp - b_exp;
        match (a_neg, b_neg) {
            (false, false) => div_floor(exp, a_m, b_m),
            (true, false) => -div_ceil(exp, a_m, b_m),
            (false, true) => -div_floor(exp, a_m, b_m),
            (true, true) => div_ceil(exp, a_m, b_m),
        }
    } else {
        // `a` or `b` are +-0.0 or infinity or NaN.
        // `a / b` is also +-0.0 or infinity or NaN.
        // There is no need to round to an integer.
        a / b
    }
}

/// Returns `floor((a << exp) / b)`.
///
/// Requires `2^10 <= a, b < 2^11`.
fn div_floor(exp: i16, a: u16, b: u16) -> f16 {
    if exp < 0 {
        0.0
    } else if exp <= 5 {
        // aa < (2^11 << 5) = 2^16
        let aa = a << exp;
        // q < 2^16 / 2^10 = 2^6
        let q = (aa / b) as u8;
        // We have to use `as` because `From<u8> for f16` is not yet implemented.
        q as f16
    } else if exp <= 16 {
        // aa < (2^11 << 16) = 2^27
        let aa = u32::from(a) << exp;
        let bb = u32::from(b);
        // q < 2^27 / 2^10 = 2^17
        let q = aa / bb;
        q as f16
    } else {
        // exp >= 17
        // result >= (2^10 << 17) / 2^11 = 2^16
        // Exponent 16 is too large to represent in f16.
        f16::INFINITY
    }
}

/// Returns `ceil((a << exp) / b)`.
///
/// Requires `2^10 <= a, b < 2^11`.
fn div_ceil(exp: i16, a: u16, b: u16) -> f16 {
    if exp < 0 {
        1.0
    } else if exp <= 5 {
        // aa < (2^11 << 5) = 2^16
        let aa = a << exp;
        // q < 2^16 / 2^10 + 1 = 2^6 + 1
        let q = ((aa - 1) / b) as u8 + 1;
        // We have to use `as` because `From<u8> for f16` is not yet implemented.
        q as f16
    } else if exp <= 16 {
        // aa <= ((2^11 - 1) << 16) = 2^27 - 2^16
        let aa = u32::from(a) << exp;
        let bb = u32::from(b);
        // q <= (2^27 - 2^16) / 2^10 + 1 = 2^17 - 2^6 + 1
        let q = ((aa - 1) / bb) as u16 + 1;
        q as f16
    } else {
        // exp >= 17
        // result >= (2^10 << 17) / 2^11 = 2^16
        // Exponent 16 is too large to represent in f16.
        f16::INFINITY
    }
}

/// For finite, non-zero numbers returns `(sign, exponent, mantissa)`.
///
/// `x = (-1)^sign * 2^exp * mantissa`
///
/// `2^10 <= mantissa < 2^11`
fn normal_form(x: f16) -> Option<(bool, i16, u16)> {
    let bits = x.to_bits();
    let sign = bits >> 15 != 0;
    let biased_exponent = (bits >> 10 & 0x1f) as i16;
    let significand = bits & 0x3ff;
    match biased_exponent {
        0 if significand == 0 => {
            // 0.0
            None
        }
        0 => {
            // Subnormal number: 2^(-14-10) * significand.
            // We want mantissa to have exactly 5 leading zeros.
            let shift = significand.leading_zeros() - 5;
            Some((sign, -14 - 10 - shift as i16, significand << shift))
        }
        0x1f => {
            // Infinity or NaN.
            None
        }
        _ => {
            // Normal number: 2^(biased_exponent-15-10) * (2^10 + significand)
            Some((sign, biased_exponent - 15 - 10, 1 << 10 | significand))
        }
    }
}
