#[cfg(test)]
mod tests;

/// Software implementation of `f32::div_euclid`.
#[allow(dead_code)]
pub(crate) fn div_euclid(a: f32, b: f32) -> f32 {
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
/// Requires `2^23 <= a, b < 2^24`
fn div_floor(exp: i32, a: u32, b: u32) -> f32 {
    if exp < 0 {
        0.0
    } else if exp <= 8 {
        // aa < (2^24 << 8) = 2^32
        let aa = a << exp;
        // q < 2^32 / 2^23 = 2^9
        let q = (aa / b) as u16;
        q.into()
    } else if exp <= 31 {
        // aa < (2^24 << 31) = 2^55
        let aa = u64::from(a) << exp;
        let bb = u64::from(b);
        // q < 2^55 / 2^23 = 2^32
        let q = (aa / bb) as u32;
        q as f32
    } else {
        // aa >= (2^23 << 31) = 2^54
        // aa < (2^24 << 31) = 2^55
        let aa = u64::from(a) << 31;
        let bb = u64::from(b);
        // e > 0
        // The result is floor((aa << e) / b).
        let e = (exp - 31) as u32;

        // aa = q * b + r
        // q >= 2^54 / 2^24 = 2^30
        // q < 2^55 / 2^23 = 2^32
        let q = (aa / bb) as u32;
        // 0 <= r < b
        let r = (aa % bb) as u32;

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
        let case_1_bound = if e < 32 { (b - 1) >> e } else { 0 };
        let q_adj = if r <= case_1_bound {
            // Case 1.
            q
        } else {
            // Case 2.
            q | 1
        };
        q_adj as f32 * pow2(e)
    }
}

/// Returns ceil((a << exp) / b).
///
/// Requires `2^23 <= a, b < 2^24`
fn div_ceil(exp: i32, a: u32, b: u32) -> f32 {
    if exp < 0 {
        1.0
    } else if exp <= 8 {
        // aa < (2^24 << 8) = 2^32
        let aa = a << exp;
        // q < 2^32 / 2^23 + 1 = 2^9 + 1
        let q = ((aa - 1) / b) as u16 + 1;
        q.into()
    } else if exp <= 31 {
        // aa <= ((2^24 - 1) << 31) = 2^55 - 2^31
        let aa = u64::from(a) << exp;
        let bb = u64::from(b);
        // q <= (2^55 - 2^31) / 2^23 + 1 = 2^32 - 2^8 + 1
        let q = ((aa - 1) / bb) as u32 + 1;
        q as f32
    } else {
        // aa >= (2^23 << 31) = 2^54
        // aa <= ((2^24 - 1) << 31) = 2^55 - 2^31
        let aa = u64::from(a) << 31;
        let bb = u64::from(b);
        // e > 0
        // The result is ceil((aa << e) / b).
        let e = (exp - 31) as u32;

        // aa = q * b + r
        // q >= 2^54 / 2^24 = 2^30
        // q <= (2^55 - 2^31) / 2^23 = 2^32 - 2^8
        let q = (aa / bb) as u32;
        // 0 <= r < b
        let r = (aa % bb) as u32;

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
        // the 25 most significant bits of q depends only on whether the low-order
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
        let case_3_bound = b - if e < 32 { (b - 1) >> e } else { 0 };
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
        q_adj as f32 * pow2(e)
    }
}

/// For finite, non-zero numbers returns `(sign, exponent, mantissa)`.
///
/// `x = (-1)^sign * 2^exp * mantissa`
///
/// `2^23 <= mantissa < 2^24`
fn normal_form(x: f32) -> Option<(bool, i32, u32)> {
    let bits = x.to_bits();
    let sign = bits >> 31 != 0;
    let biased_exponent = (bits >> 23 & 0xff) as i32;
    let significand = bits & 0x7fffff;
    match biased_exponent {
        0 if significand == 0 => {
            // 0.0
            None
        }
        0 => {
            // Subnormal number: 2^(-126-23) * significand.
            // We want mantissa to have exactly 8 leading zeros.
            let shift = significand.leading_zeros() - 8;
            Some((sign, -126 - 23 - shift as i32, significand << shift))
        }
        0xff => {
            // Infinity or NaN.
            None
        }
        _ => {
            // Normal number: 2^(biased_exponent-127-23) * (2^23 + significand)
            Some((sign, biased_exponent - 127 - 23, 1 << 23 | significand))
        }
    }
}

/// Returns `2^exp`.
fn pow2(exp: u32) -> f32 {
    if exp <= 127 { f32::from_bits((exp + 127) << 23) } else { f32::INFINITY }
}
