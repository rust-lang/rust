//! Converting decimal strings into IEEE 754 binary floating point numbers.
//!
//! # Problem statement
//!
//! We are given a decimal string such as `12.34e56`. This string consists of integral (`12`),
//! fractional (`34`), and exponent (`56`) parts. All parts are optional and interpreted as a
//! default value (1 or 0) when missing.
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
//! a type with 64 bit significand). The extended-precision algorithm
//! uses the Eisel-Lemire algorithm, which uses a 128-bit (or 192-bit)
//! representation that can accurately and quickly compute the vast majority
//! of floats. When all these fail, we bite the bullet and resort to using
//! a large-decimal representation, shifting the digits into range, calculating
//! the upper significant bits and exactly round to the nearest representation.
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
//! Primarily, this module and its children implement the algorithms described in:
//! "Number Parsing at a Gigabyte per Second", available online:
//! <https://arxiv.org/abs/2101.11408>.
//!
//! # Other
//!
//! The conversion should *never* panic. There are assertions and explicit panics in the code,
//! but they should never be triggered and only serve as internal sanity checks. Any panics should
//! be considered a bug.
//!
//! There are unit tests but they are woefully inadequate at ensuring correctness, they only cover
//! a small percentage of possible errors. Far more extensive tests are located in the directory
//! `src/etc/test-float-parse` as a Rust program.
//!
//! A note on integer overflow: Many parts of this file perform arithmetic with the decimal
//! exponent `e`. Primarily, we shift the decimal point around: Before the first decimal digit,
//! after the last decimal digit, and so on. This could overflow if done carelessly. We rely on
//! the parsing submodule to only hand out sufficiently small exponents, where "sufficient" means
//! "such that the exponent +/- the number of decimal digits fits into a 64 bit integer".
//! Larger exponents are accepted, but we don't do arithmetic with them, they are immediately
//! turned into {positive,negative} {zero,infinity}.
//!
//! # Notation
//!
//! This module uses the same notation as the Lemire paper:
//!
//! - `m`: binary mantissa; always nonnegative
//! - `p`: binary exponent; a signed integer
//! - `w`: decimal significand; always nonnegative
//! - `q`: decimal exponent; a signed integer
//!
//! This gives `m * 2^p` for the binary floating-point number, with `w * 10^q` as the decimal
//! equivalent.

#![doc(hidden)]
#![unstable(
    feature = "dec2flt",
    reason = "internal routines only exposed for testing",
    issue = "none"
)]

use self::common::BiasedFp;
use self::float::RawFloat;
use self::lemire::compute_float;
use self::parse::{parse_inf_nan, parse_number};
use self::slow::parse_long_mantissa;
use crate::error::Error;
use crate::fmt;
use crate::str::FromStr;

mod common;
pub mod decimal;
pub mod decimal_seq;
mod fpu;
mod slow;
mod table;
// float is used in flt2dec, and all are used in unit tests.
pub mod float;
pub mod lemire;
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
            /// * '5.'
            /// * '.5', or, equivalently, '0.5'
            /// * 'inf', '-inf', '+infinity', 'NaN'
            ///
            /// Note that alphabetical characters are not case-sensitive.
            ///
            /// Leading and trailing whitespace represent an error.
            ///
            /// # Grammar
            ///
            /// All strings that adhere to the following [EBNF] grammar when
            /// lowercased will result in an [`Ok`] being returned:
            ///
            /// ```txt
            /// Float  ::= Sign? ( 'inf' | 'infinity' | 'nan' | Number )
            /// Number ::= ( Digit+ |
            ///              Digit+ '.' Digit* |
            ///              Digit* '.' Digit+ ) Exp?
            /// Exp    ::= 'e' Sign? Digit+
            /// Sign   ::= [+-]
            /// Digit  ::= [0-9]
            /// ```
            ///
            /// [EBNF]: https://www.w3.org/TR/REC-xml/#sec-notation
            ///
            /// # Arguments
            ///
            /// * src - A string
            ///
            /// # Return value
            ///
            /// `Err(ParseFloatError)` if the string did not represent a valid
            /// number. Otherwise, `Ok(n)` where `n` is the closest
            /// representable floating-point number to the number represented
            /// by `src` (following the same rules for rounding as for the
            /// results of primitive operations).
            // We add the `#[inline(never)]` attribute, since its content will
            // be filled with that of `dec2flt`, which has #[inline(always)].
            // Since `dec2flt` is generic, a normal inline attribute on this function
            // with `dec2flt` having no attributes results in heavily repeated
            // generation of `dec2flt`, despite the fact only a maximum of 2
            // possible instances can ever exist. Adding #[inline(never)] avoids this.
            #[inline(never)]
            fn from_str(src: &str) -> Result<Self, ParseFloatError> {
                dec2flt(src)
            }
        }
    };
}
from_str_float_impl!(f32);
from_str_float_impl!(f64);

/// An error which can be returned when parsing a float.
///
/// This error is used as the error type for the [`FromStr`] implementation
/// for [`f32`] and [`f64`].
///
/// # Example
///
/// ```
/// use std::str::FromStr;
///
/// if let Err(e) = f64::from_str("a.12") {
///     println!("Failed conversion to f64: {e}");
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ParseFloatError {
    kind: FloatErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FloatErrorKind {
    Empty,
    Invalid,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for ParseFloatError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        match self.kind {
            FloatErrorKind::Empty => "cannot parse float from empty string",
            FloatErrorKind::Invalid => "invalid float literal",
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(deprecated)]
        self.description().fmt(f)
    }
}

#[inline]
pub(super) fn pfe_empty() -> ParseFloatError {
    ParseFloatError { kind: FloatErrorKind::Empty }
}

// Used in unit tests, keep public.
// This is much better than making FloatErrorKind and ParseFloatError::kind public.
#[inline]
pub fn pfe_invalid() -> ParseFloatError {
    ParseFloatError { kind: FloatErrorKind::Invalid }
}

/// Converts a `BiasedFp` to the closest machine float type.
fn biased_fp_to_float<F: RawFloat>(x: BiasedFp) -> F {
    let mut word = x.m;
    word |= (x.p_biased as u64) << F::SIG_BITS;
    F::from_u64_bits(word)
}

/// Converts a decimal string into a floating point number.
#[inline(always)] // Will be inlined into a function with `#[inline(never)]`, see above
pub fn dec2flt<F: RawFloat>(s: &str) -> Result<F, ParseFloatError> {
    let mut s = s.as_bytes();
    let c = if let Some(&c) = s.first() {
        c
    } else {
        return Err(pfe_empty());
    };
    let negative = c == b'-';
    if c == b'-' || c == b'+' {
        s = &s[1..];
    }
    if s.is_empty() {
        return Err(pfe_invalid());
    }

    let mut num = match parse_number(s) {
        Some(r) => r,
        None if let Some(value) = parse_inf_nan(s, negative) => return Ok(value),
        None => return Err(pfe_invalid()),
    };
    num.negative = negative;
    if !cfg!(feature = "optimize_for_size") {
        if let Some(value) = num.try_fast_path::<F>() {
            return Ok(value);
        }
    }

    // If significant digits were truncated, then we can have rounding error
    // only if `mantissa + 1` produces a different result. We also avoid
    // redundantly using the Eisel-Lemire algorithm if it was unable to
    // correctly round on the first pass.
    let mut fp = compute_float::<F>(num.exponent, num.mantissa);
    if num.many_digits
        && fp.p_biased >= 0
        && fp != compute_float::<F>(num.exponent, num.mantissa + 1)
    {
        fp.p_biased = -1;
    }
    // Unable to correctly round the float using the Eisel-Lemire algorithm.
    // Fallback to a slower, but always correct algorithm.
    if fp.p_biased < 0 {
        fp = parse_long_mantissa::<F>(s);
    }

    let mut float = biased_fp_to_float::<F>(fp);
    if num.negative {
        float = -float;
    }
    Ok(float)
}
