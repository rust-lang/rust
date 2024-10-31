//! Utilities for working with hex float formats.

#![allow(dead_code)] // FIXME: remove once this gets used

/// Construct a 32-bit float from hex float representation (C-style)
pub const fn hf32(s: &str) -> f32 {
    f32_from_bits(parse_any(s, 32, 23) as u32)
}

/// Construct a 64-bit float from hex float representation (C-style)
pub const fn hf64(s: &str) -> f64 {
    f64_from_bits(parse_any(s, 64, 52) as u64)
}

const fn parse_any(s: &str, bits: u32, sig_bits: u32) -> u128 {
    let exp_bits: u32 = bits - sig_bits - 1;
    let max_msb: i32 = (1 << (exp_bits - 1)) - 1;
    // The exponent of one ULP in the subnormals
    let min_lsb: i32 = 1 - max_msb - sig_bits as i32;

    let (neg, mut sig, exp) = parse_hex(s.as_bytes());

    if sig == 0 {
        return (neg as u128) << (bits - 1);
    }

    // exponents of the least and most significant bits in the value
    let lsb = sig.trailing_zeros() as i32;
    let msb = u128_ilog2(sig) as i32;
    let sig_bits = sig_bits as i32;

    assert!(msb - lsb <= sig_bits, "the value is too precise");
    assert!(msb + exp <= max_msb, "the value is too huge");
    assert!(lsb + exp >= min_lsb, "the value is too tiny");

    // The parsed value is X = sig * 2^exp
    // Expressed as a multiple U of the smallest subnormal value:
    // X = U * 2^min_lsb, so U = sig * 2^(exp-min_lsb)
    let mut uexp = exp - min_lsb;

    let shift = if uexp + msb >= sig_bits {
        // normal, shift msb to position sig_bits
        sig_bits - msb
    } else {
        // subnormal, shift so that uexp becomes 0
        uexp
    };

    if shift >= 0 {
        sig <<= shift;
    } else {
        sig >>= -shift;
    }
    uexp -= shift;

    // the most significant bit is like having 1 in the exponent bits
    // add any leftover exponent to that
    assert!(uexp >= 0 && uexp < (1 << exp_bits) - 2);
    sig += (uexp as u128) << sig_bits;

    // finally, set the sign bit if necessary
    sig | ((neg as u128) << (bits - 1))
}

/// Parse a hexadecimal float x
/// returns (s,n,e):
///     s == x.is_sign_negative()
///     n * 2^e == x.abs()
const fn parse_hex(mut b: &[u8]) -> (bool, u128, i32) {
    let mut neg = false;
    let mut sig: u128 = 0;
    let mut exp: i32 = 0;

    if let &[c @ (b'-' | b'+'), ref rest @ ..] = b {
        b = rest;
        neg = c == b'-';
    }

    if let &[b'0', b'x' | b'X', ref rest @ ..] = b {
        b = rest;
    } else {
        panic!("no hex indicator");
    }

    let mut seen_point = false;
    let mut some_digits = false;

    while let &[c, ref rest @ ..] = b {
        b = rest;

        match c {
            b'.' => {
                assert!(!seen_point);
                seen_point = true;
                continue;
            }
            b'p' | b'P' => break,
            c => {
                let digit = hex_digit(c);
                some_digits = true;
                let of;
                (sig, of) = sig.overflowing_mul(16);
                assert!(!of, "too many digits");
                sig |= digit as u128;
                // up until the fractional point, the value grows
                // with more digits, but after it the exponent is
                // compensated to match.
                if seen_point {
                    exp -= 4;
                }
            }
        }
    }
    assert!(some_digits, "at least one digit is required");
    some_digits = false;

    let mut negate_exp = false;
    if let &[c @ (b'-' | b'+'), ref rest @ ..] = b {
        b = rest;
        negate_exp = c == b'-';
    }

    let mut pexp: i32 = 0;
    while let &[c, ref rest @ ..] = b {
        b = rest;
        let digit = dec_digit(c);
        some_digits = true;
        let of;
        (pexp, of) = pexp.overflowing_mul(10);
        assert!(!of, "too many exponent digits");
        pexp += digit as i32;
    }
    assert!(some_digits, "at least one exponent digit is required");

    if negate_exp {
        exp -= pexp;
    } else {
        exp += pexp;
    }

    (neg, sig, exp)
}

const fn dec_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        _ => panic!("bad char"),
    }
}

const fn hex_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("bad char"),
    }
}

/* FIXME(msrv): vendor some things that are not const stable at our MSRV */

/// `f32::from_bits`
const fn f32_from_bits(v: u32) -> f32 {
    unsafe { core::mem::transmute(v) }
}

/// `f64::from_bits`
const fn f64_from_bits(v: u64) -> f64 {
    unsafe { core::mem::transmute(v) }
}

/// `u128::ilog2`
const fn u128_ilog2(v: u128) -> u32 {
    assert!(v != 0);
    u128::BITS - 1 - v.leading_zeros()
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::{format, println};

    use super::*;

    #[test]
    fn test_parse_any() {
        for k in -149..=127 {
            let s = format!("0x1p{k}");
            let x = hf32(&s);
            let y = if k < 0 { 0.5f32.powi(-k) } else { 2.0f32.powi(k) };
            assert_eq!(x, y);
        }

        let mut s = *b"0x.0000000p-121";
        for e in 0..40 {
            for k in 0..(1 << 15) {
                let expected = f32::from_bits(k) * 2.0f32.powi(e);
                let x = hf32(std::str::from_utf8(&s).unwrap());
                assert_eq!(
                    x.to_bits(),
                    expected.to_bits(),
                    "\
                    e={e}\n\
                    k={k}\n\
                    x={x}\n\
                    expected={expected}\n\
                    s={}\n\
                    f32::from_bits(k)={}\n\
                    2.0f32.powi(e)={}\
                    ",
                    std::str::from_utf8(&s).unwrap(),
                    f32::from_bits(k),
                    2.0f32.powi(e),
                );
                for i in (3..10).rev() {
                    if s[i] == b'f' {
                        s[i] = b'0';
                    } else if s[i] == b'9' {
                        s[i] = b'a';
                        break;
                    } else {
                        s[i] += 1;
                        break;
                    }
                }
            }
            for i in (12..15).rev() {
                if s[i] == b'0' {
                    s[i] = b'9';
                } else {
                    s[i] -= 1;
                    break;
                }
            }
            for i in (3..10).rev() {
                s[i] = b'0';
            }
        }
    }

    #[test]
    fn test_f32() {
        let checks = [
            ("0x.1234p+16", (0x1234 as f32).to_bits()),
            ("0x1.234p+12", (0x1234 as f32).to_bits()),
            ("0x12.34p+8", (0x1234 as f32).to_bits()),
            ("0x123.4p+4", (0x1234 as f32).to_bits()),
            ("0x1234p+0", (0x1234 as f32).to_bits()),
            ("0x1234.p+0", (0x1234 as f32).to_bits()),
            ("0x1234.0p+0", (0x1234 as f32).to_bits()),
            ("0x1.fffffep+127", f32::MAX.to_bits()),
            ("0x1.0p+1", 2.0f32.to_bits()),
            ("0x1.0p+0", 1.0f32.to_bits()),
            ("0x1.ffep+8", 0x43fff000),
            ("+0x1.ffep+8", 0x43fff000),
            ("0x1p+0", 0x3f800000),
            ("0x1.99999ap-4", 0x3dcccccd),
            ("0x1.9p+6", 0x42c80000),
            ("0x1.2d5ed2p+20", 0x4996af69),
            ("-0x1.348eb8p+10", 0xc49a475c),
            ("-0x1.33dcfep-33", 0xaf19ee7f),
            ("0x0.0p0", 0.0f32.to_bits()),
            ("-0x0.0p0", (-0.0f32).to_bits()),
            ("0x1.0p0", 1.0f32.to_bits()),
            ("0x1.99999ap-4", (0.1f32).to_bits()),
            ("-0x1.99999ap-4", (-0.1f32).to_bits()),
            ("0x1.111114p-127", 0x00444445),
            ("0x1.23456p-130", 0x00091a2b),
            ("0x1p-149", 0x00000001),
        ];
        for (s, exp) in checks {
            println!("parsing {s}");
            let act = hf32(s).to_bits();
            assert_eq!(
                act, exp,
                "parsing {s}: {act:#010x} != {exp:#010x}\nact: {act:#034b}\nexp: {exp:#034b}"
            );
        }
    }

    #[test]
    fn test_f64() {
        let checks = [
            ("0x.1234p+16", (0x1234 as f64).to_bits()),
            ("0x1.234p+12", (0x1234 as f64).to_bits()),
            ("0x12.34p+8", (0x1234 as f64).to_bits()),
            ("0x123.4p+4", (0x1234 as f64).to_bits()),
            ("0x1234p+0", (0x1234 as f64).to_bits()),
            ("0x1234.p+0", (0x1234 as f64).to_bits()),
            ("0x1234.0p+0", (0x1234 as f64).to_bits()),
            ("0x1.ffep+8", 0x407ffe0000000000),
            ("0x1p+0", 0x3ff0000000000000),
            ("0x1.999999999999ap-4", 0x3fb999999999999a),
            ("0x1.9p+6", 0x4059000000000000),
            ("0x1.2d5ed1fe1da7bp+20", 0x4132d5ed1fe1da7b),
            ("-0x1.348eb851eb852p+10", 0xc09348eb851eb852),
            ("-0x1.33dcfe54a3803p-33", 0xbde33dcfe54a3803),
            ("0x1.0p0", 1.0f64.to_bits()),
            ("0x0.0p0", 0.0f64.to_bits()),
            ("-0x0.0p0", (-0.0f64).to_bits()),
            ("0x1.999999999999ap-4", 0.1f64.to_bits()),
            ("0x1.999999999998ap-4", (0.1f64 - f64::EPSILON).to_bits()),
            ("-0x1.999999999999ap-4", (-0.1f64).to_bits()),
            ("-0x1.999999999998ap-4", (-0.1f64 + f64::EPSILON).to_bits()),
            ("0x0.8000000000001p-1022", 0x0008000000000001),
            ("0x0.123456789abcdp-1022", 0x000123456789abcd),
            ("0x0.0000000000002p-1022", 0x0000000000000002),
        ];
        for (s, exp) in checks {
            println!("parsing {s}");
            let act = hf64(s).to_bits();
            assert_eq!(
                act, exp,
                "parsing {s}: {act:#018x} != {exp:#018x}\nact: {act:#066b}\nexp: {exp:#066b}"
            );
        }
    }

    #[test]
    fn test_f32_almost_extra_precision() {
        // Exact maximum precision allowed
        hf32("0x1.abcdeep+0");
    }

    #[test]
    fn test_macros() {
        assert_eq!(hf32!("0x1.ffep+8").to_bits(), 0x43fff000u32);
        assert_eq!(hf64!("0x1.ffep+8").to_bits(), 0x407ffe0000000000u64);
    }
}

#[cfg(test)]
// FIXME(ppc): something with `should_panic` tests cause a SIGILL with ppc64le
#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
mod tests_panicking {
    extern crate std;
    use super::*;

    #[test]
    #[should_panic]
    fn test_f32_extra_precision2() {
        // One bit more than the above.
        hf32("0x1.ffffffp+127");
    }

    #[test]
    #[should_panic(expected = "the value is too huge")]
    fn test_f32_overflow() {
        // One bit more than the above.
        hf32("0x1p+128");
    }

    #[test]
    #[should_panic(expected = "the value is too precise")]
    fn test_f32_extra_precision() {
        // One bit more than the above.
        hf32("0x1.abcdefp+0");
    }

    #[test]
    fn test_f32_tiniest() {
        let x = hf32("0x1.p-149");
        let y = hf32("0x0.0000000000000001p-85");
        let z = hf32("0x0.8p-148");
        assert_eq!(x, y);
        assert_eq!(x, z);
    }

    #[test]
    #[should_panic(expected = "the value is too tiny")]
    fn test_f32_too_tiny() {
        hf32("0x1.p-150");
    }

    #[test]
    #[should_panic(expected = "the value is too tiny")]
    fn test_f32_also_too_tiny() {
        hf32("0x0.8p-149");
    }

    #[test]
    #[should_panic(expected = "the value is too tiny")]
    fn test_f32_again_too_tiny() {
        hf32("0x0.0000000000000001p-86");
    }

    #[test]
    fn test_f64_almost_extra_precision() {
        // Exact maximum precision allowed
        hf64("0x1.abcdabcdabcdfp+0");
    }

    #[test]
    #[should_panic(expected = "the value is too precise")]
    fn test_f64_extra_precision() {
        // One bit more than the above.
        hf64("0x1.abcdabcdabcdf8p+0");
    }
}
