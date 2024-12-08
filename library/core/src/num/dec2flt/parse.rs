//! Functions to parse floating-point numbers.

use crate::num::dec2flt::common::{ByteSlice, is_8digits};
use crate::num::dec2flt::float::RawFloat;
use crate::num::dec2flt::number::Number;

const MIN_19DIGIT_INT: u64 = 100_0000_0000_0000_0000;

/// Parse 8 digits, loaded as bytes in little-endian order.
///
/// This uses the trick where every digit is in [0x030, 0x39],
/// and therefore can be parsed in 3 multiplications, much
/// faster than the normal 8.
///
/// This is based off the algorithm described in "Fast numeric string to
/// int", available here: <https://johnnylee-sde.github.io/Fast-numeric-string-to-int/>.
fn parse_8digits(mut v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064;
    const MUL2: u64 = 0x0000_2710_0000_0001;
    v -= 0x3030_3030_3030_3030;
    v = (v * 10) + (v >> 8); // will not overflow, fits in 63 bits
    let v1 = (v & MASK).wrapping_mul(MUL1);
    let v2 = ((v >> 16) & MASK).wrapping_mul(MUL2);
    ((v1.wrapping_add(v2) >> 32) as u32) as u64
}

/// Parse digits until a non-digit character is found.
fn try_parse_digits(mut s: &[u8], mut x: u64) -> (&[u8], u64) {
    // may cause overflows, to be handled later

    while s.len() >= 8 {
        let num = s.read_u64();
        if is_8digits(num) {
            x = x.wrapping_mul(1_0000_0000).wrapping_add(parse_8digits(num));
            s = &s[8..];
        } else {
            break;
        }
    }

    s = s.parse_digits(|digit| {
        x = x.wrapping_mul(10).wrapping_add(digit as _);
    });

    (s, x)
}

/// Parse up to 19 digits (the max that can be stored in a 64-bit integer).
fn try_parse_19digits(s_ref: &mut &[u8], x: &mut u64) {
    let mut s = *s_ref;

    while *x < MIN_19DIGIT_INT {
        if let Some((c, s_next)) = s.split_first() {
            let digit = c.wrapping_sub(b'0');

            if digit < 10 {
                *x = (*x * 10) + digit as u64; // no overflows here
                s = s_next;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    *s_ref = s;
}

/// Parse the scientific notation component of a float.
fn parse_scientific(s_ref: &mut &[u8]) -> Option<i64> {
    let mut exponent = 0i64;
    let mut negative = false;

    let mut s = *s_ref;

    if let Some((&c, s_next)) = s.split_first() {
        negative = c == b'-';
        if c == b'-' || c == b'+' {
            s = s_next;
        }
    }

    if matches!(s.first(), Some(&x) if x.is_ascii_digit()) {
        *s_ref = s.parse_digits(|digit| {
            // no overflows here, saturate well before overflow
            if exponent < 0x10000 {
                exponent = 10 * exponent + digit as i64;
            }
        });
        if negative { Some(-exponent) } else { Some(exponent) }
    } else {
        *s_ref = s;
        None
    }
}

/// Parse a partial, non-special floating point number.
///
/// This creates a representation of the float as the
/// significant digits and the decimal exponent.
fn parse_partial_number(mut s: &[u8]) -> Option<(Number, usize)> {
    debug_assert!(!s.is_empty());

    // parse initial digits before dot
    let mut mantissa = 0_u64;
    let start = s;
    let tmp = try_parse_digits(s, mantissa);
    s = tmp.0;
    mantissa = tmp.1;
    let mut n_digits = s.offset_from(start);

    // handle dot with the following digits
    let mut n_after_dot = 0;
    let mut exponent = 0_i64;
    let int_end = s;

    if let Some((&b'.', s_next)) = s.split_first() {
        s = s_next;
        let before = s;
        let tmp = try_parse_digits(s, mantissa);
        s = tmp.0;
        mantissa = tmp.1;
        n_after_dot = s.offset_from(before);
        exponent = -n_after_dot as i64;
    }

    n_digits += n_after_dot;
    if n_digits == 0 {
        return None;
    }

    // handle scientific format
    let mut exp_number = 0_i64;
    if let Some((&c, s_next)) = s.split_first() {
        if c == b'e' || c == b'E' {
            s = s_next;
            // If None, we have no trailing digits after exponent, or an invalid float.
            exp_number = parse_scientific(&mut s)?;
            exponent += exp_number;
        }
    }

    let len = s.offset_from(start) as _;

    // handle uncommon case with many digits
    if n_digits <= 19 {
        return Some((Number { exponent, mantissa, negative: false, many_digits: false }, len));
    }

    n_digits -= 19;
    let mut many_digits = false;
    let mut p = start;
    while let Some((&c, p_next)) = p.split_first() {
        if c == b'.' || c == b'0' {
            n_digits -= c.saturating_sub(b'0' - 1) as isize;
            p = p_next;
        } else {
            break;
        }
    }
    if n_digits > 0 {
        // at this point we have more than 19 significant digits, let's try again
        many_digits = true;
        mantissa = 0;
        let mut s = start;
        try_parse_19digits(&mut s, &mut mantissa);
        exponent = if mantissa >= MIN_19DIGIT_INT {
            // big int
            int_end.offset_from(s)
        } else {
            s = &s[1..];
            let before = s;
            try_parse_19digits(&mut s, &mut mantissa);
            -s.offset_from(before)
        } as i64;
        // add back the explicit part
        exponent += exp_number;
    }

    Some((Number { exponent, mantissa, negative: false, many_digits }, len))
}

/// Try to parse a non-special floating point number,
/// as well as two slices with integer and fractional parts
/// and the parsed exponent.
pub fn parse_number(s: &[u8]) -> Option<Number> {
    if let Some((float, rest)) = parse_partial_number(s) {
        if rest == s.len() {
            return Some(float);
        }
    }
    None
}

/// Try to parse a special, non-finite float.
pub(crate) fn parse_inf_nan<F: RawFloat>(s: &[u8], negative: bool) -> Option<F> {
    // Since a valid string has at most the length 8, we can load
    // all relevant characters into a u64 and work from there.
    // This also generates much better code.

    let mut register;
    let len: usize;

    // All valid strings are either of length 8 or 3.
    if s.len() == 8 {
        register = s.read_u64();
        len = 8;
    } else if s.len() == 3 {
        let a = s[0] as u64;
        let b = s[1] as u64;
        let c = s[2] as u64;
        register = (c << 16) | (b << 8) | a;
        len = 3;
    } else {
        return None;
    }

    // Clear out the bits which turn ASCII uppercase characters into
    // lowercase characters. The resulting string is all uppercase.
    // What happens to other characters is irrelevant.
    register &= 0xDFDFDFDFDFDFDFDF;

    // u64 values corresponding to relevant cases
    const INF_3: u64 = 0x464E49; // "INF"
    const INF_8: u64 = 0x5954494E49464E49; // "INFINITY"
    const NAN: u64 = 0x4E414E; // "NAN"

    // Match register value to constant to parse string.
    // Also match on the string length to catch edge cases
    // like "inf\0\0\0\0\0".
    let float = match (register, len) {
        (INF_3, 3) => F::INFINITY,
        (INF_8, 8) => F::INFINITY,
        (NAN, 3) => F::NAN,
        _ => return None,
    };

    if negative { Some(-float) } else { Some(float) }
}
