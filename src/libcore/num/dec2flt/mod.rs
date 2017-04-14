// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Converting decimal strings into IEEE 754 binary floating point numbers.
//!
//! # Problem statement
//!
//! We are given a decimal string such as `12.34e56`. This string consists of integral (`12`),
//! fractional (`45`), and exponent (`56`) parts. All parts are optional and interpreted as zero
//! when missing.
//!
//! We seek the IEEE 754 floating point number that is closest to the exact value of the decimal
//! string. It is well-known that many decimal strings do not have terminating representations in
//! base two, so we round to 0.5 units in the last place (in other words, as well as possible).
//! Ties, decimal values exactly half-way between two consecutive floats, are resolved with the
//! half-to-even strategy, also known as banker's rounding.
//!
//! Needless to say, this is quite hard, both in terms of implementation complexity and in terms
//! of CPU cycles taken.
//!
//! # Implementation
//!
//! First, we ignore signs. Or rather, we remove it at the very beginning of the conversion
//! process and re-apply it at the very end. This is correct in all edge cases since IEEE
//! floats are symmetric around zero, negating one simply flips the first bit.
//!
//! Then we remove the decimal point by adjusting the exponent: Conceptually, `12.34e56` turns
//! into `1234e54`, which we describe with a positive integer `f = 1234` and an integer `e = 54`.
//! The `(f, e)` representation is used by almost all code past the parsing stage.
//!
//! We then try a long chain of progressively more general and expensive special cases using
//! machine-sized integers and small, fixed-sized floating point numbers (first `f32`/`f64`, then
//! a type with 64 bit significand, `Fp`). When all these fail, we bite the bullet and resort to a
//! simple but very slow algorithm that involved computing `f * 10^e` fully and doing an iterative
//! search for the best approximation.
//!
//! Primarily, this module and its children implement the algorithms described in:
//! "How to Read Floating Point Numbers Accurately" by William D. Clinger,
//! available online: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4152
//!
//! In addition, there are numerous helper functions that are used in the paper but not available
//! in Rust (or at least in core). Our version is additionally complicated by the need to handle
//! overflow and underflow and the desire to handle subnormal numbers.  Bellerophon and
//! Algorithm R have trouble with overflow, subnormals, and underflow. We conservatively switch to
//! Algorithm M (with the modifications described in section 8 of the paper) well before the
//! inputs get into the critical region.
//!
//! Another aspect that needs attention is the ``RawFloat`` trait by which almost all functions
//! are parametrized. One might think that it's enough to parse to `f64` and cast the result to
//! `f32`. Unfortunately this is not the world we live in, and this has nothing to do with using
//! base two or half-to-even rounding.
//!
//! Consider for example two types `d2` and `d4` representing a decimal type with two decimal
//! digits and four decimal digits each and take "0.01499" as input. Let's use half-up rounding.
//! Going directly to two decimal digits gives `0.01`, but if we round to four digits first,
//! we get `0.0150`, which is then rounded up to `0.02`. The same principle applies to other
//! operations as well, if you want 0.5 ULP accuracy you need to do *everything* in full precision
//! and round *exactly once, at the end*, by considering all truncated bits at once.
//!
//! FIXME Although some code duplication is necessary, perhaps parts of the code could be shuffled
//! around such that less code is duplicated. Large parts of the algorithms are independent of the
//! float type to output, or only needs access to a few constants, which could be passed in as
//! parameters.
//!
//! # Other
//!
//! The conversion should *never* panic. There are assertions and explicit panics in the code,
//! but they should never be triggered and only serve as internal sanity checks. Any panics should
//! be considered a bug.
//!
//! There are unit tests but they are woefully inadequate at ensuring correctness, they only cover
//! a small percentage of possible errors. Far more extensive tests are located in the directory
//! `src/etc/test-float-parse` as a Python script.
//!
//! A note on integer overflow: Many parts of this file perform arithmetic with the decimal
//! exponent `e`. Primarily, we shift the decimal point around: Before the first decimal digit,
//! after the last decimal digit, and so on. This could overflow if done carelessly. We rely on
//! the parsing submodule to only hand out sufficiently small exponents, where "sufficient" means
//! "such that the exponent +/- the number of decimal digits fits into a 64 bit integer".
//! Larger exponents are accepted, but we don't do arithmetic with them, they are immediately
//! turned into {positive,negative} {zero,infinity}.

#![doc(hidden)]
#![unstable(feature = "dec2flt",
            reason = "internal routines only exposed for testing",
            issue = "0")]

use fmt;
use str::FromStr;

use self::parse::{parse_decimal, Decimal, Sign, ParseResult};
use self::num::digits_to_big;
use self::rawfp::RawFloat;

mod algorithm;
mod table;
mod num;
// These two have their own tests.
pub mod rawfp;
pub mod parse;

macro_rules! from_str_float_impl {
    ($t:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl FromStr for $t {
            type Err = ParseFloatError;

            /// Converts a string in base 10 to a float.
            /// Accepts an optional decimal exponent.
            ///
            /// This function accepts strings such as
            ///
            /// * '3.14'
            /// * '-3.14'
            /// * '2.5E10', or equivalently, '2.5e10'
            /// * '2.5E-10'
            /// * '.' (understood as 0)
            /// * '5.'
            /// * '.5', or, equivalently,  '0.5'
            /// * 'inf', '-inf', 'NaN'
            ///
            /// Leading and trailing whitespace represent an error.
            ///
            /// # Arguments
            ///
            /// * src - A string
            ///
            /// # Return value
            ///
            /// `Err(ParseFloatError)` if the string did not represent a valid
            /// number.  Otherwise, `Ok(n)` where `n` is the floating-point
            /// number represented by `src`.
            #[inline]
            fn from_str(src: &str) -> Result<Self, ParseFloatError> {
                dec2flt(src)
            }
        }
    }
}
from_str_float_impl!(f32);
from_str_float_impl!(f64);

/// An error which can be returned when parsing a float.
///
/// This error is used as the error type for the [`FromStr`] implementation
/// for [`f32`] and [`f64`].
///
/// [`FromStr`]: ../str/trait.FromStr.html
/// [`f32`]: ../../std/primitive.f32.html
/// [`f64`]: ../../std/primitive.f64.html
#[derive(Debug, Clone, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ParseFloatError {
    kind: FloatErrorKind
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FloatErrorKind {
    Empty,
    Invalid,
}

impl ParseFloatError {
    #[unstable(feature = "int_error_internals",
               reason = "available through Error trait and this method should \
                         not be exposed publicly",
               issue = "0")]
    #[doc(hidden)]
    pub fn __description(&self) -> &str {
        match self.kind {
            FloatErrorKind::Empty => "cannot parse float from empty string",
            FloatErrorKind::Invalid => "invalid float literal",
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseFloatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.__description().fmt(f)
    }
}

fn pfe_empty() -> ParseFloatError {
    ParseFloatError { kind: FloatErrorKind::Empty }
}

fn pfe_invalid() -> ParseFloatError {
    ParseFloatError { kind: FloatErrorKind::Invalid }
}

/// Split decimal string into sign and the rest, without inspecting or validating the rest.
fn extract_sign(s: &str) -> (Sign, &str) {
    match s.as_bytes()[0] {
        b'+' => (Sign::Positive, &s[1..]),
        b'-' => (Sign::Negative, &s[1..]),
        // If the string is invalid, we never use the sign, so we don't need to validate here.
        _ => (Sign::Positive, s),
    }
}

/// Convert a decimal string into a floating point number.
fn dec2flt<T: RawFloat>(s: &str) -> Result<T, ParseFloatError> {
    if s.is_empty() {
        return Err(pfe_empty())
    }
    let (sign, s) = extract_sign(s);
    let flt = match parse_decimal(s) {
        ParseResult::Valid(decimal) => convert(decimal)?,
        ParseResult::ShortcutToInf => T::INFINITY,
        ParseResult::ShortcutToZero => T::ZERO,
        ParseResult::Invalid => match s {
            "inf" => T::INFINITY,
            "NaN" => T::NAN,
            _ => { return Err(pfe_invalid()); }
        }
    };

    match sign {
        Sign::Positive => Ok(flt),
        Sign::Negative => Ok(-flt),
    }
}

/// The main workhorse for the decimal-to-float conversion: Orchestrate all the preprocessing
/// and figure out which algorithm should do the actual conversion.
fn convert<T: RawFloat>(mut decimal: Decimal) -> Result<T, ParseFloatError> {
    simplify(&mut decimal);
    if let Some(x) = trivial_cases(&decimal) {
        return Ok(x);
    }
    // Remove/shift out the decimal point.
    let e = decimal.exp - decimal.fractional.len() as i64;
    if let Some(x) = algorithm::fast_path(decimal.integral, decimal.fractional, e) {
        return Ok(x);
    }
    // Big32x40 is limited to 1280 bits, which translates to about 385 decimal digits.
    // If we exceed this, we'll crash, so we error out before getting too close (within 10^10).
    let upper_bound = bound_intermediate_digits(&decimal, e);
    if upper_bound > 375 {
        return Err(pfe_invalid());
    }
    let f = digits_to_big(decimal.integral, decimal.fractional);

    // Now the exponent certainly fits in 16 bit, which is used throughout the main algorithms.
    let e = e as i16;
    // FIXME These bounds are rather conservative. A more careful analysis of the failure modes
    // of Bellerophon could allow using it in more cases for a massive speed up.
    let exponent_in_range = table::MIN_E <= e && e <= table::MAX_E;
    let value_in_range = upper_bound <= T::MAX_NORMAL_DIGITS as u64;
    if exponent_in_range && value_in_range {
        Ok(algorithm::bellerophon(&f, e))
    } else {
        Ok(algorithm::algorithm_m(&f, e))
    }
}

// As written, this optimizes badly (see #27130, though it refers to an old version of the code).
// `inline(always)` is a workaround for that. There are only two call sites overall and it doesn't
// make code size worse.

/// Strip zeros where possible, even when this requires changing the exponent
#[inline(always)]
fn simplify(decimal: &mut Decimal) {
    let is_zero = &|&&d: &&u8| -> bool { d == b'0' };
    // Trimming these zeros does not change anything but may enable the fast path (< 15 digits).
    let leading_zeros = decimal.integral.iter().take_while(is_zero).count();
    decimal.integral = &decimal.integral[leading_zeros..];
    let trailing_zeros = decimal.fractional.iter().rev().take_while(is_zero).count();
    let end = decimal.fractional.len() - trailing_zeros;
    decimal.fractional = &decimal.fractional[..end];
    // Simplify numbers of the form 0.0...x and x...0.0, adjusting the exponent accordingly.
    // This may not always be a win (possibly pushes some numbers out of the fast path), but it
    // simplifies other parts significantly (notably, approximating the magnitude of the value).
    if decimal.integral.is_empty() {
        let leading_zeros = decimal.fractional.iter().take_while(is_zero).count();
        decimal.fractional = &decimal.fractional[leading_zeros..];
        decimal.exp -= leading_zeros as i64;
    } else if decimal.fractional.is_empty() {
        let trailing_zeros = decimal.integral.iter().rev().take_while(is_zero).count();
        let end = decimal.integral.len() - trailing_zeros;
        decimal.integral = &decimal.integral[..end];
        decimal.exp += trailing_zeros as i64;
    }
}

/// Quick and dirty upper bound on the size (log10) of the largest value that Algorithm R and
/// Algorithm M will compute while working on the given decimal.
fn bound_intermediate_digits(decimal: &Decimal, e: i64) -> u64 {
    // We don't need to worry too much about overflow here thanks to trivial_cases() and the
    // parser, which filter out the most extreme inputs for us.
    let f_len: u64 = decimal.integral.len() as u64 + decimal.fractional.len() as u64;
    if e >= 0 {
        // In the case e >= 0, both algorithms compute about `f * 10^e`. Algorithm R proceeds to
        // do some complicated calculations with this but we can ignore that for the upper bound
        // because it also reduces the fraction beforehand, so we have plenty of buffer there.
        f_len + (e as u64)
    } else {
        // If e < 0, Algorithm R does roughly the same thing, but Algorithm M differs:
        // It tries to find a positive number k such that `f << k / 10^e` is an in-range
        // significand. This will result in about `2^53 * f * 10^e` < `10^17 * f * 10^e`.
        // One input that triggers this is 0.33...33 (375 x 3).
        f_len + (e.abs() as u64) + 17
    }
}

/// Detect obvious overflows and underflows without even looking at the decimal digits.
fn trivial_cases<T: RawFloat>(decimal: &Decimal) -> Option<T> {
    // There were zeros but they were stripped by simplify()
    if decimal.integral.is_empty() && decimal.fractional.is_empty() {
        return Some(T::ZERO);
    }
    // This is a crude approximation of ceil(log10(the real value)). We don't need to worry too
    // much about overflow here because the input length is tiny (at least compared to 2^64) and
    // the parser already handles exponents whose absolute value is greater than 10^18
    // (which is still 10^19 short of 2^64).
    let max_place = decimal.exp + decimal.integral.len() as i64;
    if max_place > T::INF_CUTOFF {
        return Some(T::INFINITY);
    } else if max_place < T::ZERO_CUTOFF {
        return Some(T::ZERO);
    }
    None
}
