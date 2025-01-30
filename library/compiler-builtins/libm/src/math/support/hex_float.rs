//! Utilities for working with hex float formats.

#![allow(dead_code)] // FIXME: remove once this gets used

use core::fmt;

use super::{Float, f32_from_bits, f64_from_bits};

/// Construct a 16-bit float from hex float representation (C-style)
#[cfg(f16_enabled)]
pub const fn hf16(s: &str) -> f16 {
    f16::from_bits(parse_any(s, 16, 10) as u16)
}

/// Construct a 32-bit float from hex float representation (C-style)
pub const fn hf32(s: &str) -> f32 {
    f32_from_bits(parse_any(s, 32, 23) as u32)
}

/// Construct a 64-bit float from hex float representation (C-style)
pub const fn hf64(s: &str) -> f64 {
    f64_from_bits(parse_any(s, 64, 52) as u64)
}

/// Construct a 128-bit float from hex float representation (C-style)
#[cfg(f128_enabled)]
pub const fn hf128(s: &str) -> f128 {
    f128::from_bits(parse_any(s, 128, 112))
}

/// Parse any float from hex to its bitwise representation.
///
/// `nan_repr` is passed rather than constructed so the platform-specific NaN is returned.
pub const fn parse_any(s: &str, bits: u32, sig_bits: u32) -> u128 {
    let exp_bits: u32 = bits - sig_bits - 1;
    let max_msb: i32 = (1 << (exp_bits - 1)) - 1;
    // The exponent of one ULP in the subnormals
    let min_lsb: i32 = 1 - max_msb - sig_bits as i32;

    let exp_mask = ((1 << exp_bits) - 1) << sig_bits;

    let (neg, mut sig, exp) = match parse_hex(s.as_bytes()) {
        Parsed::Finite { neg, sig: 0, .. } => return (neg as u128) << (bits - 1),
        Parsed::Finite { neg, sig, exp } => (neg, sig, exp),
        Parsed::Infinite { neg } => return ((neg as u128) << (bits - 1)) | exp_mask,
        Parsed::Nan { neg } => {
            return ((neg as u128) << (bits - 1)) | exp_mask | (1 << (sig_bits - 1));
        }
    };

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

/// A parsed floating point number.
enum Parsed {
    /// Absolute value sig * 2^e
    Finite {
        neg: bool,
        sig: u128,
        exp: i32,
    },
    Infinite {
        neg: bool,
    },
    Nan {
        neg: bool,
    },
}

/// Parse a hexadecimal float x
const fn parse_hex(mut b: &[u8]) -> Parsed {
    let mut neg = false;
    let mut sig: u128 = 0;
    let mut exp: i32 = 0;

    if let &[c @ (b'-' | b'+'), ref rest @ ..] = b {
        b = rest;
        neg = c == b'-';
    }

    match *b {
        [b'i' | b'I', b'n' | b'N', b'f' | b'F'] => return Parsed::Infinite { neg },
        [b'n' | b'N', b'a' | b'A', b'n' | b'N'] => return Parsed::Nan { neg },
        _ => (),
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

    Parsed::Finite { neg, sig, exp }
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

/// `u128::ilog2`
const fn u128_ilog2(v: u128) -> u32 {
    assert!(v != 0);
    u128::BITS - 1 - v.leading_zeros()
}

/// Format a floating point number as its IEEE hex (`%a`) representation.
pub struct Hexf<F>(pub F);

// Adapted from https://github.com/ericseppanen/hexfloat2/blob/a5c27932f0ff/src/format.rs
fn fmt_any_hex<F: Float>(x: &F, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if x.is_sign_negative() {
        write!(f, "-")?;
    }

    if x.is_nan() {
        return write!(f, "NaN");
    } else if x.is_infinite() {
        return write!(f, "inf");
    } else if *x == F::ZERO {
        return write!(f, "0x0p+0");
    }

    let mut exponent = x.exp_unbiased();
    let sig = x.to_bits() & F::SIG_MASK;

    let bias = F::EXP_BIAS as i32;
    // The mantissa MSB needs to be shifted up to the nearest nibble.
    let mshift = (4 - (F::SIG_BITS % 4)) % 4;
    let sig = sig << mshift;
    // The width is rounded up to the nearest char (4 bits)
    let mwidth = (F::SIG_BITS as usize + 3) / 4;
    let leading = if exponent == -bias {
        // subnormal number means we shift our output by 1 bit.
        exponent += 1;
        "0."
    } else {
        "1."
    };

    write!(f, "0x{leading}{sig:0mwidth$x}p{exponent:+}")
}

#[cfg(f16_enabled)]
impl fmt::LowerHex for Hexf<f16> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_any_hex(&self.0, f)
    }
}

impl fmt::LowerHex for Hexf<f32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_any_hex(&self.0, f)
    }
}

impl fmt::LowerHex for Hexf<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_any_hex(&self.0, f)
    }
}

#[cfg(f128_enabled)]
impl fmt::LowerHex for Hexf<f128> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_any_hex(&self.0, f)
    }
}

impl fmt::LowerHex for Hexf<i32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl<T1, T2> fmt::LowerHex for Hexf<(T1, T2)>
where
    T1: Copy,
    T2: Copy,
    Hexf<T1>: fmt::LowerHex,
    Hexf<T2>: fmt::LowerHex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:x}, {:x})", Hexf(self.0.0), Hexf(self.0.1))
    }
}

impl<T> fmt::Debug for Hexf<T>
where
    Hexf<T>: fmt::LowerHex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(self, f)
    }
}

impl<T> fmt::Display for Hexf<T>
where
    Hexf<T>: fmt::LowerHex,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(self, f)
    }
}

#[cfg(test)]
mod parse_tests {
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

    // HACK(msrv): 1.63 rejects unknown width float literals at an AST level, so use a macro to
    // hide them from the AST.
    #[cfg(f16_enabled)]
    macro_rules! f16_tests {
        () => {
            #[test]
            fn test_f16() {
                let checks = [
                    ("0x.1234p+16", (0x1234 as f16).to_bits()),
                    ("0x1.234p+12", (0x1234 as f16).to_bits()),
                    ("0x12.34p+8", (0x1234 as f16).to_bits()),
                    ("0x123.4p+4", (0x1234 as f16).to_bits()),
                    ("0x1234p+0", (0x1234 as f16).to_bits()),
                    ("0x1234.p+0", (0x1234 as f16).to_bits()),
                    ("0x1234.0p+0", (0x1234 as f16).to_bits()),
                    ("0x1.ffcp+15", f16::MAX.to_bits()),
                    ("0x1.0p+1", 2.0f16.to_bits()),
                    ("0x1.0p+0", 1.0f16.to_bits()),
                    ("0x1.ffp+8", 0x5ffc),
                    ("+0x1.ffp+8", 0x5ffc),
                    ("0x1p+0", 0x3c00),
                    ("0x1.998p-4", 0x2e66),
                    ("0x1.9p+6", 0x5640),
                    ("0x0.0p0", 0.0f16.to_bits()),
                    ("-0x0.0p0", (-0.0f16).to_bits()),
                    ("0x1.0p0", 1.0f16.to_bits()),
                    ("0x1.998p-4", (0.1f16).to_bits()),
                    ("-0x1.998p-4", (-0.1f16).to_bits()),
                    ("0x0.123p-12", 0x0123),
                    ("0x1p-24", 0x0001),
                    ("nan", f16::NAN.to_bits()),
                    ("-nan", (-f16::NAN).to_bits()),
                    ("inf", f16::INFINITY.to_bits()),
                    ("-inf", f16::NEG_INFINITY.to_bits()),
                ];
                for (s, exp) in checks {
                    println!("parsing {s}");
                    let act = hf16(s).to_bits();
                    assert_eq!(
                        act, exp,
                        "parsing {s}: {act:#06x} != {exp:#06x}\nact: {act:#018b}\nexp: {exp:#018b}"
                    );
                }
            }

            #[test]
            fn test_macros_f16() {
                assert_eq!(hf16!("0x1.ffp+8").to_bits(), 0x5ffc_u16);
            }
        };
    }

    #[cfg(f16_enabled)]
    f16_tests!();

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
            ("nan", f32::NAN.to_bits()),
            ("-nan", (-f32::NAN).to_bits()),
            ("inf", f32::INFINITY.to_bits()),
            ("-inf", f32::NEG_INFINITY.to_bits()),
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
            ("nan", f64::NAN.to_bits()),
            ("-nan", (-f64::NAN).to_bits()),
            ("inf", f64::INFINITY.to_bits()),
            ("-inf", f64::NEG_INFINITY.to_bits()),
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

    // HACK(msrv): 1.63 rejects unknown width float literals at an AST level, so use a macro to
    // hide them from the AST.
    #[cfg(f128_enabled)]
    macro_rules! f128_tests {
        () => {
            #[test]
            fn test_f128() {
                let checks = [
                    ("0x.1234p+16", (0x1234 as f128).to_bits()),
                    ("0x1.234p+12", (0x1234 as f128).to_bits()),
                    ("0x12.34p+8", (0x1234 as f128).to_bits()),
                    ("0x123.4p+4", (0x1234 as f128).to_bits()),
                    ("0x1234p+0", (0x1234 as f128).to_bits()),
                    ("0x1234.p+0", (0x1234 as f128).to_bits()),
                    ("0x1234.0p+0", (0x1234 as f128).to_bits()),
                    ("0x1.ffffffffffffffffffffffffffffp+16383", f128::MAX.to_bits()),
                    ("0x1.0p+1", 2.0f128.to_bits()),
                    ("0x1.0p+0", 1.0f128.to_bits()),
                    ("0x1.ffep+8", 0x4007ffe0000000000000000000000000),
                    ("+0x1.ffep+8", 0x4007ffe0000000000000000000000000),
                    ("0x1p+0", 0x3fff0000000000000000000000000000),
                    ("0x1.999999999999999999999999999ap-4", 0x3ffb999999999999999999999999999a),
                    ("0x1.9p+6", 0x40059000000000000000000000000000),
                    ("0x0.0p0", 0.0f128.to_bits()),
                    ("-0x0.0p0", (-0.0f128).to_bits()),
                    ("0x1.0p0", 1.0f128.to_bits()),
                    ("0x1.999999999999999999999999999ap-4", (0.1f128).to_bits()),
                    ("-0x1.999999999999999999999999999ap-4", (-0.1f128).to_bits()),
                    ("0x0.abcdef0123456789abcdef012345p-16382", 0x0000abcdef0123456789abcdef012345),
                    ("0x1p-16494", 0x00000000000000000000000000000001),
                    ("nan", f128::NAN.to_bits()),
                    ("-nan", (-f128::NAN).to_bits()),
                    ("inf", f128::INFINITY.to_bits()),
                    ("-inf", f128::NEG_INFINITY.to_bits()),
                ];
                for (s, exp) in checks {
                    println!("parsing {s}");
                    let act = hf128(s).to_bits();
                    assert_eq!(
                        act, exp,
                        "parsing {s}: {act:#034x} != {exp:#034x}\nact: {act:#0130b}\nexp: {exp:#0130b}"
                    );
                }
            }

            #[test]
            fn test_macros_f128() {
                assert_eq!(hf128!("0x1.ffep+8").to_bits(), 0x4007ffe0000000000000000000000000_u128);
            }
        }
    }

    #[cfg(f128_enabled)]
    f128_tests!();

    #[test]
    fn test_macros() {
        // FIXME(msrv): enable once parsing works
        // #[cfg(f16_enabled)]
        // assert_eq!(hf16!("0x1.ffp+8").to_bits(), 0x5ffc_u16);
        assert_eq!(hf32!("0x1.ffep+8").to_bits(), 0x43fff000_u32);
        assert_eq!(hf64!("0x1.ffep+8").to_bits(), 0x407ffe0000000000_u64);
        // FIXME(msrv): enable once parsing works
        // #[cfg(f128_enabled)]
        // assert_eq!(hf128!("0x1.ffep+8").to_bits(), 0x4007ffe0000000000000000000000000_u128);
    }
}

#[cfg(test)]
// FIXME(ppc): something with `should_panic` tests cause a SIGILL with ppc64le
#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
mod tests_panicking {
    extern crate std;
    use super::*;

    // HACK(msrv): 1.63 rejects unknown width float literals at an AST level, so use a macro to
    // hide them from the AST.
    #[cfg(f16_enabled)]
    macro_rules! f16_tests {
        () => {
            #[test]
            fn test_f16_almost_extra_precision() {
                // Exact maximum precision allowed
                hf16("0x1.ffcp+0");
            }

            #[test]
            #[should_panic(expected = "the value is too precise")]
            fn test_f16_extra_precision() {
                // One bit more than the above.
                hf16("0x1.ffdp+0");
            }

            #[test]
            #[should_panic(expected = "the value is too huge")]
            fn test_f16_overflow() {
                // One bit more than the above.
                hf16("0x1p+16");
            }

            #[test]
            fn test_f16_tiniest() {
                let x = hf16("0x1.p-24");
                let y = hf16("0x0.001p-12");
                let z = hf16("0x0.8p-23");
                assert_eq!(x, y);
                assert_eq!(x, z);
            }

            #[test]
            #[should_panic(expected = "the value is too tiny")]
            fn test_f16_too_tiny() {
                hf16("0x1.p-25");
            }

            #[test]
            #[should_panic(expected = "the value is too tiny")]
            fn test_f16_also_too_tiny() {
                hf16("0x0.8p-24");
            }

            #[test]
            #[should_panic(expected = "the value is too tiny")]
            fn test_f16_again_too_tiny() {
                hf16("0x0.001p-13");
            }
        };
    }

    #[cfg(f16_enabled)]
    f16_tests!();

    #[test]
    fn test_f32_almost_extra_precision() {
        // Exact maximum precision allowed
        hf32("0x1.abcdeep+0");
    }

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

    // HACK(msrv): 1.63 rejects unknown width float literals at an AST level, so use a macro to
    // hide them from the AST.
    #[cfg(f128_enabled)]
    macro_rules! f128_tests {
        () => {
            #[test]
            fn test_f128_almost_extra_precision() {
                // Exact maximum precision allowed
                hf128("0x1.ffffffffffffffffffffffffffffp+16383");
            }

            #[test]
            #[should_panic(expected = "the value is too precise")]
            fn test_f128_extra_precision() {
                // One bit more than the above.
                hf128("0x1.ffffffffffffffffffffffffffff8p+16383");
            }

            #[test]
            #[should_panic(expected = "the value is too huge")]
            fn test_f128_overflow() {
                // One bit more than the above.
                hf128("0x1p+16384");
            }

            #[test]
            fn test_f128_tiniest() {
                let x = hf128("0x1.p-16494");
                let y = hf128("0x0.0000000000000001p-16430");
                let z = hf128("0x0.8p-16493");
                assert_eq!(x, y);
                assert_eq!(x, z);
            }

            #[test]
            #[should_panic(expected = "the value is too tiny")]
            fn test_f128_too_tiny() {
                hf128("0x1.p-16495");
            }

            #[test]
            #[should_panic(expected = "the value is too tiny")]
            fn test_f128_again_too_tiny() {
                hf128("0x0.0000000000000001p-16431");
            }

            #[test]
            #[should_panic(expected = "the value is too tiny")]
            fn test_f128_also_too_tiny() {
                hf128("0x0.8p-16494");
            }
        };
    }

    #[cfg(f128_enabled)]
    f128_tests!();
}

#[cfg(test)]
mod print_tests {
    extern crate std;
    use std::string::ToString;

    use super::*;

    #[test]
    #[cfg(f16_enabled)]
    fn test_f16() {
        use std::format;
        // Exhaustively check that `f16` roundtrips.
        for x in 0..=u16::MAX {
            let f = f16::from_bits(x);
            let s = format!("{}", Hexf(f));
            let from_s = hf16(&s);

            if f.is_nan() && from_s.is_nan() {
                continue;
            }

            assert_eq!(
                f.to_bits(),
                from_s.to_bits(),
                "{f:?} formatted as {s} but parsed as {from_s:?}"
            );
        }
    }

    #[test]
    fn spot_checks() {
        assert_eq!(Hexf(f32::MAX).to_string(), "0x1.fffffep+127");
        assert_eq!(Hexf(f64::MAX).to_string(), "0x1.fffffffffffffp+1023");

        assert_eq!(Hexf(f32::MIN).to_string(), "-0x1.fffffep+127");
        assert_eq!(Hexf(f64::MIN).to_string(), "-0x1.fffffffffffffp+1023");

        assert_eq!(Hexf(f32::ZERO).to_string(), "0x0p+0");
        assert_eq!(Hexf(f64::ZERO).to_string(), "0x0p+0");

        assert_eq!(Hexf(f32::NEG_ZERO).to_string(), "-0x0p+0");
        assert_eq!(Hexf(f64::NEG_ZERO).to_string(), "-0x0p+0");

        assert_eq!(Hexf(f32::NAN).to_string(), "NaN");
        assert_eq!(Hexf(f64::NAN).to_string(), "NaN");

        assert_eq!(Hexf(f32::INFINITY).to_string(), "inf");
        assert_eq!(Hexf(f64::INFINITY).to_string(), "inf");

        assert_eq!(Hexf(f32::NEG_INFINITY).to_string(), "-inf");
        assert_eq!(Hexf(f64::NEG_INFINITY).to_string(), "-inf");

        #[cfg(f16_enabled)]
        {
            assert_eq!(Hexf(f16::MAX).to_string(), "0x1.ffcp+15");
            assert_eq!(Hexf(f16::MIN).to_string(), "-0x1.ffcp+15");
            assert_eq!(Hexf(f16::ZERO).to_string(), "0x0p+0");
            assert_eq!(Hexf(f16::NEG_ZERO).to_string(), "-0x0p+0");
            assert_eq!(Hexf(f16::NAN).to_string(), "NaN");
            assert_eq!(Hexf(f16::INFINITY).to_string(), "inf");
            assert_eq!(Hexf(f16::NEG_INFINITY).to_string(), "-inf");
        }

        #[cfg(f128_enabled)]
        {
            assert_eq!(Hexf(f128::MAX).to_string(), "0x1.ffffffffffffffffffffffffffffp+16383");
            assert_eq!(Hexf(f128::MIN).to_string(), "-0x1.ffffffffffffffffffffffffffffp+16383");
            assert_eq!(Hexf(f128::ZERO).to_string(), "0x0p+0");
            assert_eq!(Hexf(f128::NEG_ZERO).to_string(), "-0x0p+0");
            assert_eq!(Hexf(f128::NAN).to_string(), "NaN");
            assert_eq!(Hexf(f128::INFINITY).to_string(), "inf");
            assert_eq!(Hexf(f128::NEG_INFINITY).to_string(), "-inf");
        }
    }
}
