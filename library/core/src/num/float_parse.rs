//! User-facing API for float parsing.

use crate::error::Error;
use crate::fmt;
use crate::num::imp::dec2flt;
use crate::str::FromStr;

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
            /// * '7'
            /// * '007'
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
                dec2flt::dec2flt(src)
            }
        }
    };
}

#[cfg(target_has_reliable_f16)]
from_str_float_impl!(f16);
from_str_float_impl!(f32);
from_str_float_impl!(f64);

// FIXME(f16): A fallback is used when the backend+target does not support f16 well, in order
// to avoid ICEs.

#[cfg(not(target_has_reliable_f16))]
impl FromStr for f16 {
    type Err = ParseFloatError;

    #[inline]
    fn from_str(_src: &str) -> Result<Self, ParseFloatError> {
        unimplemented!("requires target_has_reliable_f16")
    }
}

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
    pub(super) kind: FloatErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FloatErrorKind {
    Empty,
    Invalid,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for ParseFloatError {}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            FloatErrorKind::Empty => "cannot parse float from empty string",
            FloatErrorKind::Invalid => "invalid float literal",
        }
        .fmt(f)
    }
}
