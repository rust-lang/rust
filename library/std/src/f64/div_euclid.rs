#[cfg(test)]
mod tests;

/// Software implementation of `f64::div_euclid`.
#[allow(dead_code)]
pub(crate) fn div_euclid(a: f64, b: f64) -> f64 {
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
/// Requires `2^52 <= a, b < 2^53`.
fn div_floor(exp: i32, a: u64, b: u64) -> f64 {
    if exp < 0 {
        0.0
    } else if exp <= 11 {
        // aa < (2^53 << 11) = 2^64
        let aa = a << exp;
        // q < 2^64 / 2^52 = 2^12
        let q = (aa / b) as u32;
        q.into()
    } else if exp <= 63 {
        // aa < (2^53 << 63) = 2^116
        let aa = u128::from(a) << exp;
        let bb = u128::from(b);
        // q < 2^116 / 2^52 = 2^64
        let q = (aa / bb) as u64;
        q as f64
    } else {
        // aa >= (2^52 << 63) = 2^115
        // aa < (2^53 << 63) = 2^116
        let aa = u128::from(a) << 63;
        let bb = u128::from(b);
        // e > 0
        // The result is floor((aa << e) / b).
        let e = (exp - 63) as u32;

        // aa = q * b + r
        // q >= 2^115 / 2^53 = 2^62
        // q < 2^116 / 2^52 = 2^64
        let q = (aa / bb) as u64;
        // 0 <= r < b
        let r = (aa % bb) as u64;

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
        let case_1_bound = if e < 64 { (b - 1) >> e } else { 0 };
        let q_adj = if r <= case_1_bound {
            // Case 1.
            q
        } else {
            // Case 2.
            q | 1
        };
        q_adj as f64 * pow2(e)
    }
}

/// Returns `ceil((a << exp) / b)`.
///
/// Requires `2^52 <= a, b < 2^53`.
fn div_ceil(exp: i32, a: u64, b: u64) -> f64 {
    if exp < 0 {
        1.0
    } else if exp <= 11 {
        // aa < (2^53 << 11) = 2^64
        let aa = a << exp;
        // q < 2^64 / 2^52 + 1 = 2^12 + 1
        let q = ((aa - 1) / b) as u32 + 1;
        q.into()
    } else if exp <= 63 {
        // aa <= ((2^53 - 1) << 63) = 2^116 - 2^63
        let aa = u128::from(a) << exp;
        let bb = u128::from(b);
        // q <= (2^116 - 2^63) / 2^52 + 1 = 2^64 - 2^11 + 1
        let q = ((aa - 1) / bb) as u64 + 1;
        q as f64
    } else {
        // aa >= (2^52 << 63) = 2^115
        // aa <= ((2^53 - 1) << 63) = 2^116 - 2^63
        let aa = u128::from(a) << 63;
        let bb = u128::from(b);
        // e > 0
        // The result is ceil((aa << e) / b).
        let e = (exp - 63) as u32;

        // aa = q * b + r
        // q >= 2^115 / 2^53 = 2^62
        // q <= (2^116 - 2^63) / 2^52 = 2^64 - 2^11
        let q = (aa / bb) as u64;
        // 0 <= r < b
        let r = (aa % bb) as u64;

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
        let case_3_bound = b - if e < 64 { (b - 1) >> e } else { 0 };
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
        q_adj as f64 * pow2(e)
    }
}

/// For finite, non-zero numbers returns `(sign, exponent, mantissa)`.
///
/// `x = (-1)^sign * 2^exp * mantissa`
///
/// `2^52 <= mantissa < 2^53`
fn normal_form(x: f64) -> Option<(bool, i32, u64)> {
    let bits = x.to_bits();
    let sign = bits >> 63 != 0;
    let biased_exponent = (bits >> 52 & 0x7ff) as i32;
    let significand = bits & ((1 << 52) - 1);
    match biased_exponent {
        0 if significand == 0 => {
            // 0.0
            None
        }
        0 => {
            // Subnormal number: 2^(-1022-52) * significand.
            // We want mantissa to have exactly 11 leading zeros.
            let shift = significand.leading_zeros() - 11;
            Some((sign, -1022 - 52 - shift as i32, significand << shift))
        }
        0x7ff => {
            // Infinity or NaN.
            None
        }
        _ => {
            // Normal number: 2^(biased_exponent-1023-52) * (2^52 + significand)
            Some((sign, biased_exponent - 1023 - 52, 1 << 52 | significand))
        }
    }
}

/// Returns `2^exp`.
fn pow2(exp: u32) -> f64 {
    if exp <= 1023 { f64::from_bits(u64::from(exp + 1023) << 52) } else { f64::INFINITY }
}
