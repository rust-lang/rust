#[cfg(test)]
mod tests;

use crate::f128::u256::U256;

/// Software implementation of `f128::div_euclid`.
#[allow(dead_code)]
pub(crate) fn div_euclid(a: f128, b: f128) -> f128 {
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
/// Requires `2^112 <= a, b < 2^113`.
fn div_floor(exp: i32, a: u128, b: u128) -> f128 {
    if exp < 0 {
        0.0
    } else if exp <= 15 {
        // aa < (2^113 << 15) = 2^128
        let aa = a << exp;
        // q < 2^128 / 2^112 = 2^16
        let q = (aa / b) as u32;
        // We have to use `as` because `From<u32> for f128` is not yet implemented.
        q as f128
    } else if exp <= 127 {
        // aa = a << exp
        // aa < (2^113 << 127) = 2^240
        let aa = U256::shl_u128(a, exp as u32);
        // q < 2^240 / 2^112 = 2^128
        let (q, _) = aa.div_rem(b);
        q as f128
    } else {
        // aa >= (2^112 << 127) = 2^239
        // aa < (2^113 << 127) = 2^240
        let aa = U256::shl_u128(a, 127);
        // e > 0
        // The result is floor((aa << e) / b).
        let e = (exp - 127) as u32;

        // aa = q * b + r
        // q >= 2^239 / 2^113 = 2^126
        // q < 2^239 / 2^112 = 2^128
        // 0 <= r < b
        let (q, r) = aa.div_rem(b);

        // result = floor((aa << e) / b) = (q << e) + floor((r << e) / b)
        // 0 <= (r << e) / b < 2^e
        //
        // There are two cases:
        // 1. floor((r << e) / b) = 0
        // 2. 0 < floor((r << e) / b) < 2^e
        //
        // In case 1:
        // The result is q << e.
        //
        // In case 2:
        // The result is (q << e) + non-zero low e bits.
        // This rounds the same way as (q | 1) << e because rounding beyond
        // the 25 most significant bits of q depends only on whether the low-order
        // bits are non-zero.
        //
        // Case 1 happens when:
        // (r << e) / b < 1
        // (r << e) <= b - 1
        // r <= ((b - 1) >> e)
        let case_1_bound = if e < 128 { (b - 1) >> e } else { 0 };
        let q_adj = if r <= case_1_bound {
            // Case 1.
            q
        } else {
            // Case 2.
            q | 1
        };
        q_adj as f128 * pow2(e)
    }
}

/// Returns `ceil((a << exp) / b)`.
///
/// Requires `2^112 <= a, b < 2^113`.
fn div_ceil(exp: i32, a: u128, b: u128) -> f128 {
    if exp < 0 {
        1.0
    } else if exp <= 15 {
        // aa < (2^113 << 15) = 2^128
        let aa = a << exp;
        // q < 2^128 / 2^112 + 1 = 2^16 + 1
        let q = ((aa - 1) / b) as u32 + 1;
        // We have to use `as` because `From<u32> for f128` is not yet implemented.
        q as f128
    } else if exp <= 127 {
        // aa = a << exp
        // aa <= ((2^113 - 1) << 127) = 2^240 - 2^127
        let aa = U256::shl_u128(a, exp as u32);
        // q <= (2^240 - 2^127) / 2^112 + 1 = 2^128 - 2^15 + 1
        let (q, _) = (aa - U256::ONE).div_rem(b);
        (q + 1) as f128
    } else {
        // aa >= (2^112 << 127) = 2^239
        // aa <= ((2^113 - 1) << 127) = 2^240 - 2^127
        let aa = U256::shl_u128(a, 127);
        // e > 0
        // The result is ceil((aa << e) / b).
        let e = (exp - 127) as u32;

        // aa = q * b + r
        // q >= 2^239 / 2^112 = 2^126
        // q <= (2^240 - 2^127) / 2^112 = 2^128 - 2^15
        // 0 <= r < b
        let (q, r) = aa.div_rem(b);

        // result = ceil((aa << e) / b) = (q << e) + ceil((r << e) / b)
        // 0 <= (r << e) / b < 2^e
        //
        // There are three cases:
        // 1. ceil((r << e) / b) = 0
        // 2. 0 < ceil((r << e) / b) < 2^e
        // 3. ceil((r << e) / b) = 2^e
        //
        // In case 1:
        // The result is q << e.
        //
        // In case 2:
        // The result is (q << e) + non-zero low e bits.
        // This rounds the same way as (q | 1) << e because rounding beyond
        // the 54 most significant bits of q depends only on whether the low-order
        // bits are non-zero.
        //
        // In case 3:
        // The result is (q + 1) << e.
        //
        // Case 1 happens when r = 0.
        // Case 3 happens when:
        // (r << e) / b > (1 << e) - 1
        // (r << e) > (b << e) - b
        // ((b - r) << e) <= b - 1
        // b - r <= (b - 1) >> e
        // r >= b - ((b - 1) >> e)
        let case_3_bound = b - if e < 128 { (b - 1) >> e } else { 0 };
        let q_adj = if r == 0 {
            // Case 1.
            q
        } else if r < case_3_bound {
            // Case 2.
            q | 1
        } else {
            // Case 3.
            q + 1
        };
        q_adj as f128 * pow2(e)
    }
}

/// For finite, non-zero numbers returns (sign, exponent, mantissa).
///
/// `x = (-1)^sign * 2^exp * mantissa`
///
/// `2^112 <= mantissa < 2^113`
fn normal_form(x: f128) -> Option<(bool, i32, u128)> {
    let bits = x.to_bits();
    let sign = bits >> 127 != 0;
    let biased_exponent = (bits >> 112 & 0x7fff) as i32;
    let significand = bits & ((1 << 112) - 1);
    match biased_exponent {
        0 if significand == 0 => {
            // 0.0
            None
        }
        0 => {
            // Subnormal number: 2^(-16382-112) * significand.
            // We want mantissa to have exactly 15 leading zeros.
            let shift = significand.leading_zeros() - 15;
            Some((sign, -16382 - 112 - shift as i32, significand << shift))
        }
        0x7fff => {
            // Infinity or NaN.
            None
        }
        _ => {
            // Normal number: 2^(biased_exponent-16383-112) * (2^112 + significand)
            Some((sign, biased_exponent - 16383 - 112, 1 << 112 | significand))
        }
    }
}

/// Returns `2^exp`.
fn pow2(exp: u32) -> f128 {
    if exp <= 16383 { f128::from_bits(u128::from(exp + 16383) << 112) } else { f128::INFINITY }
}
