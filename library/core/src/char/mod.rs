//! Utilities for the `char` primitive type.
//!
//! *[See also the `char` primitive type](primitive@char).*
//!
//! The `char` type represents a single character. More specifically, since
//! 'character' isn't a well-defined concept in Unicode, `char` is a '[Unicode
//! scalar value]', which is similar to, but not the same as, a '[Unicode code
//! point]'.
//!
//! [Unicode scalar value]: https://www.unicode.org/glossary/#unicode_scalar_value
//! [Unicode code point]: https://www.unicode.org/glossary/#code_point
//!
//! This module exists for technical reasons, the primary documentation for
//! `char` is directly on [the `char` primitive type][char] itself.
//!
//! This module is the home of the iterator implementations for the iterators
//! implemented on `char`, as well as some useful constants and conversion
//! functions that convert various types to `char`.

#![allow(non_snake_case)]
#![stable(feature = "core_char", since = "1.2.0")]

mod convert;
mod decode;
mod methods;

// stable re-exports
#[stable(feature = "try_from", since = "1.34.0")]
pub use self::convert::CharTryFromError;
#[stable(feature = "char_from_str", since = "1.20.0")]
pub use self::convert::ParseCharError;
#[stable(feature = "decode_utf16", since = "1.9.0")]
pub use self::decode::{DecodeUtf16, DecodeUtf16Error};

// perma-unstable re-exports
#[unstable(feature = "char_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::methods::encode_utf16_raw;
#[unstable(feature = "char_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::methods::encode_utf8_raw;

use crate::ascii;
use crate::error::Error;
use crate::escape;
use crate::fmt::{self, Write};
use crate::iter::FusedIterator;
use crate::num::NonZeroUsize;

pub(crate) use self::methods::EscapeDebugExtArgs;

// UTF-8 ranges and tags for encoding characters
const TAG_CONT: u8 = 0b1000_0000;
const TAG_TWO_B: u8 = 0b1100_0000;
const TAG_THREE_B: u8 = 0b1110_0000;
const TAG_FOUR_B: u8 = 0b1111_0000;
const MAX_ONE_B: u32 = 0x80;
const MAX_TWO_B: u32 = 0x800;
const MAX_THREE_B: u32 = 0x10000;

/*
    Lu  Uppercase_Letter        an uppercase letter
    Ll  Lowercase_Letter        a lowercase letter
    Lt  Titlecase_Letter        a digraphic character, with first part uppercase
    Lm  Modifier_Letter         a modifier letter
    Lo  Other_Letter            other letters, including syllables and ideographs
    Mn  Nonspacing_Mark         a nonspacing combining mark (zero advance width)
    Mc  Spacing_Mark            a spacing combining mark (positive advance width)
    Me  Enclosing_Mark          an enclosing combining mark
    Nd  Decimal_Number          a decimal digit
    Nl  Letter_Number           a letterlike numeric character
    No  Other_Number            a numeric character of other type
    Pc  Connector_Punctuation   a connecting punctuation mark, like a tie
    Pd  Dash_Punctuation        a dash or hyphen punctuation mark
    Ps  Open_Punctuation        an opening punctuation mark (of a pair)
    Pe  Close_Punctuation       a closing punctuation mark (of a pair)
    Pi  Initial_Punctuation     an initial quotation mark
    Pf  Final_Punctuation       a final quotation mark
    Po  Other_Punctuation       a punctuation mark of other type
    Sm  Math_Symbol             a symbol of primarily mathematical use
    Sc  Currency_Symbol         a currency sign
    Sk  Modifier_Symbol         a non-letterlike modifier symbol
    So  Other_Symbol            a symbol of other type
    Zs  Space_Separator         a space character (of various non-zero widths)
    Zl  Line_Separator          U+2028 LINE SEPARATOR only
    Zp  Paragraph_Separator     U+2029 PARAGRAPH SEPARATOR only
    Cc  Control                 a C0 or C1 control code
    Cf  Format                  a format control character
    Cs  Surrogate               a surrogate code point
    Co  Private_Use             a private-use character
    Cn  Unassigned              a reserved unassigned code point or a noncharacter
*/

/// The highest valid code point a `char` can have, `'\u{10FFFF}'`. Use [`char::MAX`] instead.
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: char = char::MAX;

/// `U+FFFD REPLACEMENT CHARACTER` (�) is used in Unicode to represent a
/// decoding error. Use [`char::REPLACEMENT_CHARACTER`] instead.
#[stable(feature = "decode_utf16", since = "1.9.0")]
pub const REPLACEMENT_CHARACTER: char = char::REPLACEMENT_CHARACTER;

/// The version of [Unicode](https://www.unicode.org/) that the Unicode parts of
/// `char` and `str` methods are based on. Use [`char::UNICODE_VERSION`] instead.
#[stable(feature = "unicode_version", since = "1.45.0")]
pub const UNICODE_VERSION: (u8, u8, u8) = char::UNICODE_VERSION;

/// Creates an iterator over the UTF-16 encoded code points in `iter`, returning
/// unpaired surrogates as `Err`s. Use [`char::decode_utf16`] instead.
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[inline]
pub fn decode_utf16<I: IntoIterator<Item = u16>>(iter: I) -> DecodeUtf16<I::IntoIter> {
    self::decode::decode_utf16(iter)
}

/// Converts a `u32` to a `char`. Use [`char::from_u32`] instead.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_char_convert", since = "1.67.0")]
#[must_use]
#[inline]
pub const fn from_u32(i: u32) -> Option<char> {
    self::convert::from_u32(i)
}

/// Converts a `u32` to a `char`, ignoring validity. Use [`char::from_u32_unchecked`].
/// instead.
#[stable(feature = "char_from_unchecked", since = "1.5.0")]
#[rustc_const_unstable(feature = "const_char_from_u32_unchecked", issue = "89259")]
#[must_use]
#[inline]
pub const unsafe fn from_u32_unchecked(i: u32) -> char {
    // SAFETY: the safety contract must be upheld by the caller.
    unsafe { self::convert::from_u32_unchecked(i) }
}

/// Converts a digit in the given radix to a `char`. Use [`char::from_digit`] instead.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_char_convert", since = "1.67.0")]
#[must_use]
#[inline]
pub const fn from_digit(num: u32, radix: u32) -> Option<char> {
    self::convert::from_digit(num, radix)
}

/// Returns an iterator that yields the hexadecimal Unicode escape of a
/// character, as `char`s.
///
/// This `struct` is created by the [`escape_unicode`] method on [`char`]. See
/// its documentation for more.
///
/// [`escape_unicode`]: char::escape_unicode
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeUnicode(escape::EscapeIterInner<10>);

impl EscapeUnicode {
    fn new(chr: char) -> Self {
        let mut data = [ascii::Char::Null; 10];
        let range = escape::escape_unicode_into(&mut data, chr);
        Self(escape::EscapeIterInner::new(data, range))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EscapeUnicode {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        self.0.next().map(char::from)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.0.len()
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        self.0.next_back().map(char::from)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.0.advance_by(n)
    }
}

#[stable(feature = "exact_size_escape", since = "1.11.0")]
impl ExactSizeIterator for EscapeUnicode {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for EscapeUnicode {}

#[stable(feature = "char_struct_display", since = "1.16.0")]
impl fmt::Display for EscapeUnicode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.as_str())
    }
}

/// An iterator that yields the literal escape code of a `char`.
///
/// This `struct` is created by the [`escape_default`] method on [`char`]. See
/// its documentation for more.
///
/// [`escape_default`]: char::escape_default
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeDefault(escape::EscapeIterInner<10>);

impl EscapeDefault {
    fn printable(chr: ascii::Char) -> Self {
        let data = [chr];
        Self(escape::EscapeIterInner::from_array(data))
    }

    fn backslash(chr: ascii::Char) -> Self {
        let data = [ascii::Char::ReverseSolidus, chr];
        Self(escape::EscapeIterInner::from_array(data))
    }

    fn from_unicode(esc: EscapeUnicode) -> Self {
        Self(esc.0)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EscapeDefault {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        self.0.next().map(char::from)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.0.len()
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        self.0.next_back().map(char::from)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.0.advance_by(n)
    }
}

#[stable(feature = "exact_size_escape", since = "1.11.0")]
impl ExactSizeIterator for EscapeDefault {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for EscapeDefault {}

#[stable(feature = "char_struct_display", since = "1.16.0")]
impl fmt::Display for EscapeDefault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.as_str())
    }
}

/// An iterator that yields the literal escape code of a `char`.
///
/// This `struct` is created by the [`escape_debug`] method on [`char`]. See its
/// documentation for more.
///
/// [`escape_debug`]: char::escape_debug
#[stable(feature = "char_escape_debug", since = "1.20.0")]
#[derive(Clone, Debug)]
pub struct EscapeDebug(EscapeDebugInner);

#[derive(Clone, Debug)]
// Note: It’s possible to manually encode the EscapeDebugInner inside of
// EscapeIterInner (e.g. with alive=254..255 indicating that data[0..4] holds
// a char) which would likely result in a more optimised code.  For now we use
// the option easier to implement.
enum EscapeDebugInner {
    Bytes(escape::EscapeIterInner<10>),
    Char(char),
}

impl EscapeDebug {
    fn printable(chr: char) -> Self {
        Self(EscapeDebugInner::Char(chr))
    }

    fn backslash(chr: ascii::Char) -> Self {
        let data = [ascii::Char::ReverseSolidus, chr];
        let iter = escape::EscapeIterInner::from_array(data);
        Self(EscapeDebugInner::Bytes(iter))
    }

    fn from_unicode(esc: EscapeUnicode) -> Self {
        Self(EscapeDebugInner::Bytes(esc.0))
    }

    fn clear(&mut self) {
        let bytes = escape::EscapeIterInner::from_array([]);
        self.0 = EscapeDebugInner::Bytes(bytes);
    }
}

#[stable(feature = "char_escape_debug", since = "1.20.0")]
impl Iterator for EscapeDebug {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        match self.0 {
            EscapeDebugInner::Bytes(ref mut bytes) => bytes.next().map(char::from),
            EscapeDebugInner::Char(chr) => {
                self.clear();
                Some(chr)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

#[stable(feature = "char_escape_debug", since = "1.20.0")]
impl ExactSizeIterator for EscapeDebug {
    fn len(&self) -> usize {
        match &self.0 {
            EscapeDebugInner::Bytes(bytes) => bytes.len(),
            EscapeDebugInner::Char(_) => 1,
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for EscapeDebug {}

#[stable(feature = "char_escape_debug", since = "1.20.0")]
impl fmt::Display for EscapeDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            EscapeDebugInner::Bytes(bytes) => f.write_str(bytes.as_str()),
            EscapeDebugInner::Char(chr) => f.write_char(*chr),
        }
    }
}

/// Returns an iterator that yields the lowercase equivalent of a `char`.
///
/// This `struct` is created by the [`to_lowercase`] method on [`char`]. See
/// its documentation for more.
///
/// [`to_lowercase`]: char::to_lowercase
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug, Clone)]
pub struct ToLowercase(CaseMappingIter);

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for ToLowercase {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[stable(feature = "case_mapping_double_ended", since = "1.59.0")]
impl DoubleEndedIterator for ToLowercase {
    fn next_back(&mut self) -> Option<char> {
        self.0.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for ToLowercase {}

#[stable(feature = "exact_size_case_mapping_iter", since = "1.35.0")]
impl ExactSizeIterator for ToLowercase {}

/// Returns an iterator that yields the uppercase equivalent of a `char`.
///
/// This `struct` is created by the [`to_uppercase`] method on [`char`]. See
/// its documentation for more.
///
/// [`to_uppercase`]: char::to_uppercase
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug, Clone)]
pub struct ToUppercase(CaseMappingIter);

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for ToUppercase {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[stable(feature = "case_mapping_double_ended", since = "1.59.0")]
impl DoubleEndedIterator for ToUppercase {
    fn next_back(&mut self) -> Option<char> {
        self.0.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for ToUppercase {}

#[stable(feature = "exact_size_case_mapping_iter", since = "1.35.0")]
impl ExactSizeIterator for ToUppercase {}

#[derive(Debug, Clone)]
enum CaseMappingIter {
    Three(char, char, char),
    Two(char, char),
    One(char),
    Zero,
}

impl CaseMappingIter {
    fn new(chars: [char; 3]) -> CaseMappingIter {
        if chars[2] == '\0' {
            if chars[1] == '\0' {
                CaseMappingIter::One(chars[0]) // Including if chars[0] == '\0'
            } else {
                CaseMappingIter::Two(chars[0], chars[1])
            }
        } else {
            CaseMappingIter::Three(chars[0], chars[1], chars[2])
        }
    }
}

impl Iterator for CaseMappingIter {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        match *self {
            CaseMappingIter::Three(a, b, c) => {
                *self = CaseMappingIter::Two(b, c);
                Some(a)
            }
            CaseMappingIter::Two(b, c) => {
                *self = CaseMappingIter::One(c);
                Some(b)
            }
            CaseMappingIter::One(c) => {
                *self = CaseMappingIter::Zero;
                Some(c)
            }
            CaseMappingIter::Zero => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = match self {
            CaseMappingIter::Three(..) => 3,
            CaseMappingIter::Two(..) => 2,
            CaseMappingIter::One(_) => 1,
            CaseMappingIter::Zero => 0,
        };
        (size, Some(size))
    }
}

impl DoubleEndedIterator for CaseMappingIter {
    fn next_back(&mut self) -> Option<char> {
        match *self {
            CaseMappingIter::Three(a, b, c) => {
                *self = CaseMappingIter::Two(a, b);
                Some(c)
            }
            CaseMappingIter::Two(b, c) => {
                *self = CaseMappingIter::One(b);
                Some(c)
            }
            CaseMappingIter::One(c) => {
                *self = CaseMappingIter::Zero;
                Some(c)
            }
            CaseMappingIter::Zero => None,
        }
    }
}

impl fmt::Display for CaseMappingIter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CaseMappingIter::Three(a, b, c) => {
                f.write_char(a)?;
                f.write_char(b)?;
                f.write_char(c)
            }
            CaseMappingIter::Two(b, c) => {
                f.write_char(b)?;
                f.write_char(c)
            }
            CaseMappingIter::One(c) => f.write_char(c),
            CaseMappingIter::Zero => Ok(()),
        }
    }
}

#[stable(feature = "char_struct_display", since = "1.16.0")]
impl fmt::Display for ToLowercase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[stable(feature = "char_struct_display", since = "1.16.0")]
impl fmt::Display for ToUppercase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The error type returned when a checked char conversion fails.
#[stable(feature = "u8_from_char", since = "1.59.0")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TryFromCharError(pub(crate) ());

#[stable(feature = "u8_from_char", since = "1.59.0")]
impl fmt::Display for TryFromCharError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        "unicode code point out of range".fmt(fmt)
    }
}

#[stable(feature = "u8_from_char", since = "1.59.0")]
impl Error for TryFromCharError {}
