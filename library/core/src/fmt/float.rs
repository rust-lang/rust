use crate::fmt::{Debug, Display, Formatter, LowerExp, Result, UpperExp};
use crate::mem::MaybeUninit;
use crate::num::FpCategory;
use crate::num::flt2dec::SHORT_DIGITS_MAX;
use crate::num::flt2dec::decoder::{Decoded64, decode_f16, decode_f32, decode_f64};
use crate::num::flt2dec::strategy::{dragon, grisu};

/// The maximum return of enough_buf_for_fixed limits buffer allocation.
const FIXED_DIGITS_MAX: usize = 827;

/// Both `format_fixed` functions will fill the entire buffer. The return is an
/// upper bound to limit the formatting of trailing zeroes. F64 has a worst case
/// of FIXED_DIGITS_MAX (when exp is -1074).
fn enough_digits_for_fixed(dec: &Decoded64) -> usize {
    let e = dec.exp; // base 2
    // The exact digit count is either:
    //  (a) ⌈log₁₀(5⁻ᵉ × (2⁶⁴ − 1))⌉ when e < 0, or
    //  (b) ⌈log₁₀(2ᵉ × (2⁶⁴ − 1))⌉ when e ≥ 0.
    //
    // An upper bound is acquired by using the following approximations:
    //  ⌈log₁₀(2⁶⁴ − 1)⌉ ≤ 20
    //  ⌈e × log₁₀(5)⌉ ≤ 1 + e × log₁₀(5)
    //  ⌈e × log₁₀(2)⌉ ≤ 1 + e × log₁₀(2)
    //  log₁₀(5) < 12/16
    //  log₁₀(2) < 5/16
    let multiplier: isize = if e < 0 {
        -12 // negates e
    } else {
        5
    };
    // Division by 16 is done as a bit shift, explicitly.
    20 + 1 + ((multiplier * e) as usize >> 4)
}

/// Format to f in decimal notation given a value as "0."<digits> * 10^pow10.
/// The fractional part gets extended with trailing zeroes when the result has
/// less than min_nfrac decimal places.
fn fmt_digits(
    f: &mut Formatter<'_>,
    sign: &'static str,
    digits: &str,
    pow10: isize,
    min_nfrac: usize,
) -> Result {
    if pow10 <= 0 {
        // All of the digits are in the fractional part.
        const LEAD: &str = "0.";
        let lead_zeroes = (-pow10) as usize;
        let more_zeroes = min_nfrac.saturating_sub(lead_zeroes + digits.len());
        let out_len = LEAD.len() + lead_zeroes + digits.len() + more_zeroes;
        f.pad_number(sign, out_len, |w| {
            w.write_str(LEAD)?;
            w.write_zeroes(lead_zeroes)?;
            w.write_str(digits)?;
            w.write_zeroes(more_zeroes)
        })
    } else if (pow10 as usize) < digits.len() {
        // Split the digits into an integer and a fractional part.
        let (int, frac) = digits.as_bytes().split_at(pow10 as usize);
        let more_zeroes = min_nfrac.saturating_sub(frac.len());
        let out_len = int.len() + ".".len() + frac.len() + more_zeroes;
        f.pad_number(sign, out_len, |w| {
            let (int_str, frac_str) =
                // SAFETY: Digits contains single-byte characters, exclusively.
                unsafe { (str::from_utf8_unchecked(int), str::from_utf8_unchecked(frac)) };
            w.write_str(int_str)?;
            w.write_str(".")?;
            w.write_str(frac_str)?;
            w.write_zeroes(more_zeroes)
        })
    } else {
        // None of the digits are in the fractional part.
        let more_zeroes = pow10 as usize - digits.len();
        let frac_len = if min_nfrac == 0 { 0 } else { ".".len() + min_nfrac };
        let out_len = digits.len() + more_zeroes + frac_len;
        f.pad_number(sign, out_len, |w| {
            w.write_str(digits)?;
            w.write_zeroes(more_zeroes)?;
            if min_nfrac != 0 {
                w.write_str(".")?;
                w.write_zeroes(min_nfrac)?;
            }
            Ok(())
        })
    }
}

/// Format in decimal notation to f with at least min_nfrac decimal places, yet
/// no more than needed for an exact representation of the decoded value.
/// Decoded64 must contain a finite, non-zero floating-point.
fn fmt_short(
    f: &mut Formatter<'_>,
    sign: &'static str,
    dec: &Decoded64,
    min_nfrac: usize,
) -> Result {
    let mut buf = [MaybeUninit::<u8>::uninit(); SHORT_DIGITS_MAX];
    let (digits, pow10) = if let Some(res) = grisu::format_short(dec, &mut buf) {
        res
    } else {
        dragon::format_short(dec, &mut buf)
    };
    // The decoded number is presented as "0."<digits> * 10^pow10

    fmt_digits(f, sign, digits, pow10, min_nfrac)
}

/// Format in fixed-point notation to f with nfrac decimal places.
/// Decoded64 must contain a finite, non-zero floating-point.
#[inline(never)] // Only allocate the rather large stack-buffer when needed.
fn fmt_fixed(f: &mut Formatter<'_>, sign: &'static str, dec: &Decoded64, nfrac: usize) -> Result {
    // BUG: An excessive number of decimal places gets replaced by trailing
    // zeroes without warning.
    let mut buf = [MaybeUninit::<u8>::uninit(); FIXED_DIGITS_MAX];
    let buf_enough = &mut buf[..enough_digits_for_fixed(dec)];
    let limit = -(nfrac as isize); // buf_enough is the hard upper-bound

    let (digits, pow10) = if let Some(res) = grisu::format_fixed(dec, buf_enough, limit) {
        res
    } else {
        dragon::format_fixed(dec, &mut buf, limit)
    };
    // The decoded number is presented as "0."<digits> * 10^pow10

    if !digits.is_empty() {
        fmt_digits(f, sign, digits, pow10, nfrac)
    } else {
        // The number rounds down to zero at nfrac precision.
        if nfrac == 0 {
            f.pad_number(sign, 1, |w| w.write_str("0"))
        } else {
            const LEAD: &str = "0.";
            let out_len = LEAD.len() + nfrac;
            f.pad_number(sign, out_len, |w| {
                w.write_str(LEAD)?;
                w.write_zeroes(nfrac)
            })
        }
    }
}

/// Format in E notation to f with the least amount of decimal places needed for
/// an exact representation of the decoded the value.
/// Decoded64 must contain a finite, non-zero floating-point.
fn fmt_enote_short(
    f: &mut Formatter<'_>,
    sign: &'static str,
    dec: &Decoded64,
    letter_e: u8,
) -> Result {
    let mut buf = [MaybeUninit::<u8>::uninit(); SHORT_DIGITS_MAX];
    let (digits, pow10) = if let Some(res) = grisu::format_short(dec, &mut buf) {
        res
    } else {
        dragon::format_short(dec, &mut buf)
    };
    // The decoded number is presented as "0."<digits> * 10^pow10

    // E-notation is formatted as <first-digit>[.<remaining-digits>]<scale>.
    let (int, frac, sep_len) = if digits.len() > 1 {
        let (first, remain) = digits.as_bytes().split_at(1);
        // SAFETY: Digits contains single-byte characters, exclusively.
        unsafe { (str::from_utf8_unchecked(first), str::from_utf8_unchecked(remain), ".".len()) }
    } else {
        (digits, "", 0)
    };

    let mut scale_buf = [MaybeUninit::<u8>::uninit(); 5];
    let scale = encode_scale(&mut scale_buf, pow10 - 1, letter_e);

    let out_len = int.len() + sep_len + frac.len() + scale.len();
    f.pad_number(sign, out_len, |w| {
        w.write_str(int)?;
        if !frac.is_empty() {
            w.write_str(".")?;
            w.write_str(frac)?;
        }
        w.write_str(scale)
    })
}

/// Format in E notation to f with a fixed amount of decimal places.
/// Decoded64 must contain a finite, non-zero floating-point.
#[inline(never)] // Only allocate the rather large stack-buffer when needed.
fn fmt_enote_fixed(
    f: &mut Formatter<'_>,
    sign: &'static str,
    dec: &Decoded64,
    letter_e: u8,
    nfrac: usize,
) -> Result {
    // BUG: An excessive number of decimal places gets replaced by trailing
    // zeroes without warning.
    let mut buf = [MaybeUninit::<u8>::uninit(); FIXED_DIGITS_MAX];
    let buf_enough = &mut buf[..enough_digits_for_fixed(dec).min(nfrac + 1)];
    let (digits, pow10) = if let Some(res) = grisu::format_fixed(dec, buf_enough, isize::MIN) {
        res
    } else {
        dragon::format_fixed(dec, buf_enough, isize::MIN)
    };
    // The decoded number is presented as "0."<digits> * 10^pow10

    // E-notation is formatted as <first-digit>[.<remaining-digits>]<scale>.
    let (int, frac) = if digits.len() > 1 {
        let (first, remain) = digits.as_bytes().split_at(1);
        // SAFETY: Digits contains single-byte characters, exclusively.
        unsafe { (str::from_utf8_unchecked(first), str::from_utf8_unchecked(remain)) }
    } else {
        (digits, "")
    };

    let mut scale_buf = [MaybeUninit::<u8>::uninit(); 5];
    let scale = encode_scale(&mut scale_buf, pow10 - 1, letter_e);

    let frac_len = if nfrac == 0 { 0 } else { ".".len() + nfrac };
    let out_len = "1".len() + frac_len + scale.len();
    f.pad_number(sign, out_len, |w| {
        w.write_str(int)?;
        if nfrac != 0 {
            w.write_str(".")?;
            w.write_str(frac)?;
            w.write_zeroes(nfrac.saturating_sub(frac.len()))?;
        }
        w.write_str(scale)
    })
}

// Encode the E notation into buf and slice the result as a string.
fn encode_scale<'a>(buf: &'a mut [MaybeUninit<u8>; 5], e: isize, letter_e: u8) -> &'a str {
    assert!(letter_e == b'E' || letter_e == b'e');
    buf[0].write(letter_e);
    let dig_start = if e < 0 {
        buf[1].write(b'-');
        2
    } else {
        1
    };

    assert!(e > -1000 && e < 1000);
    let n = e.abs();
    let ndec = if n < 10 {
        buf[dig_start].write(b'0' + n as u8);
        1
    } else if n < 100 {
        buf[dig_start + 0].write(b'0' + (n / 10) as u8);
        buf[dig_start + 1].write(b'0' + (n % 10) as u8);
        2
    } else {
        buf[dig_start + 0].write(b'0' + (n / 100) as u8);
        let cent = n % 100;
        buf[dig_start + 1].write(b'0' + (cent / 10) as u8);
        buf[dig_start + 2].write(b'0' + (cent % 10) as u8);
        3
    };

    // SAFETY: All bytes up until dig_start + ndec have been set with ASCII.
    unsafe { str::from_utf8_unchecked(buf[..dig_start + ndec].assume_init_ref()) }
}

macro_rules! floating {
    ($($T:ident)*) => {
        $(

        fn ${concat(sign_for_, $T)}(f: &Formatter<'_>, v: $T) -> &'static str {
            if v.is_sign_negative() {
                "-"
            } else if f.sign_plus() {
                "+"
            } else {
                ""
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Debug for $T {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let sign: &'static str = ${concat(sign_for_, $T)}(f, *self);
                match self.classify() {
                    FpCategory::Nan => f.pad_number("", 3, | w | w.write_str("NaN")),
                    FpCategory::Infinite => f.pad_number(sign, 3, | w | w.write_str("inf")),
                    FpCategory::Zero => match f.precision() {
                        None | Some(1) => f.pad_number(sign, 3, | w | w.write_str("0.0")),
                        Some(0) => f.pad_number(sign, 1, | w | w.write_str("0")),
                        Some(n) => f.pad_number(sign, "0.".len() + n, | w | {
                            w.write_str("0.")?;
                            w.write_zeroes(n)
                        }),
                    },
                    FpCategory::Subnormal | FpCategory::Normal => {
                        let dec = ${concat(decode_, $T)}(*self);

                        // The appliance of precision predates the LowerExp mode
                        // for big and small values, as done next. On the other
                        // hand, Debug does declare output as “not stable”. FIX?
                        if let Some(n) = f.precision() {
                            return fmt_fixed(f, sign, &dec, n.into());
                        }

                        // Use E notation for small and large values.
                        let abs = self.abs();
                        if abs < 1e-4 || abs >= 1e+16 {
                            fmt_enote_short(f, sign, &dec, b'e')
                        } else {
                            fmt_short(f, sign, &dec, 1)
                        }
                    },
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Display for $T {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let sign: &'static str = ${concat(sign_for_, $T)}(f, *self);
                match self.classify() {
                    FpCategory::Nan => f.pad_number("", 3, | w | w.write_str("NaN")),
                    FpCategory::Infinite => f.pad_number(sign, 3, | w | w.write_str("inf")),
                    FpCategory::Zero => match f.precision() {
                        None | Some(0) => f.pad_number(sign, 1, | w | w.write_str("0")),
                        Some(n) => f.pad_number(sign, "0.".len() + n, | w | {
                            w.write_str("0.")?;
                            w.write_zeroes(n)
                        }),
                    },
                    FpCategory::Subnormal | FpCategory::Normal => {
                        let dec = ${concat(decode_, $T)}(*self);
                        if let Some(n) = f.precision() {
                            fmt_fixed(f, sign, &dec, n.into())
                        } else {
                            fmt_short(f, sign, &dec, 0)
                        }
                    },
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl LowerExp for $T {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                ${concat(fmt_exp_, $T)}(f, *self, b'e')
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl UpperExp for $T {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                ${concat(fmt_exp_, $T)}(f, *self, b'E')
            }
        }

        fn ${concat(fmt_exp_, $T)}(f: &mut Formatter<'_>, v: $T, letter_e: u8) -> Result {
            let sign: &'static str = ${concat(sign_for_, $T)}(f, v);
            match v.classify() {
                FpCategory::Nan => f.pad_number("", 3, | w | w.write_str("NaN")),
                FpCategory::Infinite => f.pad_number(sign, 3, | w | w.write_str("inf")),
                FpCategory::Zero => {
                    let fix = if letter_e == b'E' { "0E0" } else { "0e0" };
                    match f.precision() {
                        None | Some(0) => f.pad_number(sign, fix.len(), | w | w.write_str(fix)),
                        Some(n) => f.pad_number(sign, "0.".len() + n + "E0".len(), | w | {
                            w.write_str("0.")?;
                            w.write_zeroes(n - 1)?;
                            w.write_str(&fix)
                        }),
                    }
                },
                FpCategory::Subnormal | FpCategory::Normal => {
                    let dec = ${concat(decode_, $T)}(v);
                    if let Some(n) = f.precision() {
                        fmt_enote_fixed(f, sign, &dec, letter_e, n)
                    } else {
                        fmt_enote_short(f, sign, &dec, letter_e)
                    }
                },
            }
        }

        )*
    };
}

floating! { f32 f64 }

#[cfg(target_has_reliable_f16)]
floating! { f16 }

// FIXME(f16_f128): A fallback is used when the backend+target does not support f16 well, in order
// to avoid ICEs.

#[cfg(not(target_has_reliable_f16))]
#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for f16 {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:#06x}", self.to_bits())
    }
}

#[cfg(not(target_has_reliable_f16))]
#[stable(feature = "rust1", since = "1.0.0")]
impl Display for f16 {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        Debug::fmt(self, fmt)
    }
}

#[cfg(not(target_has_reliable_f16))]
#[stable(feature = "rust1", since = "1.0.0")]
impl LowerExp for f16 {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        Debug::fmt(self, fmt)
    }
}

#[cfg(not(target_has_reliable_f16))]
#[stable(feature = "rust1", since = "1.0.0")]
impl UpperExp for f16 {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        Debug::fmt(self, fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for f128 {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:#034x}", self.to_bits())
    }
}
