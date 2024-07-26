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
#[rustfmt::skip]
#[stable(feature = "try_from", since = "1.34.0")]
pub use self::convert::CharTryFromError;
#[stable(feature = "char_from_str", since = "1.20.0")]
pub use self::convert::ParseCharError;
#[stable(feature = "decode_utf16", since = "1.9.0")]
pub use self::decode::{DecodeUtf16, DecodeUtf16Error};

// perma-unstable re-exports
#[rustfmt::skip]
#[unstable(feature = "char_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::methods::encode_utf16_raw; // perma-unstable
#[unstable(feature = "char_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::methods::encode_utf8_raw; // perma-unstable

#[rustfmt::skip]
use crate::ascii;
use crate::error::Error;
use crate::escape;
use crate::fmt::{self, Write};
use crate::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use crate::num::NonZero;

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
#[rustc_const_stable(feature = "const_char_from_u32_unchecked", since = "CURRENT_RUSTC_VERSION")]
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
    #[inline]
    const fn new(c: char) -> Self {
        Self(escape::EscapeIterInner::unicode(c))
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
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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
    #[inline]
    const fn printable(c: ascii::Char) -> Self {
        Self(escape::EscapeIterInner::ascii(c.to_u8()))
    }

    #[inline]
    const fn backslash(c: ascii::Char) -> Self {
        Self(escape::EscapeIterInner::backslash(c))
    }

    #[inline]
    const fn unicode(c: char) -> Self {
        Self(escape::EscapeIterInner::unicode(c))
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
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
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
    #[inline]
    const fn printable(chr: char) -> Self {
        Self(EscapeDebugInner::Char(chr))
    }

    #[inline]
    const fn backslash(c: ascii::Char) -> Self {
        Self(EscapeDebugInner::Bytes(escape::EscapeIterInner::backslash(c)))
    }

    #[inline]
    const fn unicode(c: char) -> Self {
        Self(EscapeDebugInner::Bytes(escape::EscapeIterInner::unicode(c)))
    }

    #[inline]
    fn clear(&mut self) {
        self.0 = EscapeDebugInner::Bytes(escape::EscapeIterInner::empty());
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

    #[inline]
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

macro_rules! casemappingiter_impls {
    ($(#[$attr:meta])* $ITER_NAME:ident) => {
        $(#[$attr])*
        #[stable(feature = "rust1", since = "1.0.0")]
        #[derive(Debug, Clone)]
        pub struct $ITER_NAME(CaseMappingIter);

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Iterator for $ITER_NAME {
            type Item = char;
            fn next(&mut self) -> Option<char> {
                self.0.next()
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }

            fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
            where
                Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                self.0.fold(init, fold)
            }

            fn count(self) -> usize {
                self.0.count()
            }

            fn last(self) -> Option<Self::Item> {
                self.0.last()
            }

            fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
                self.0.advance_by(n)
            }

            unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
                // SAFETY: just forwarding requirements to caller
                unsafe { self.0.__iterator_get_unchecked(idx) }
            }
        }

        #[stable(feature = "case_mapping_double_ended", since = "1.59.0")]
        impl DoubleEndedIterator for $ITER_NAME {
            fn next_back(&mut self) -> Option<char> {
                self.0.next_back()
            }

            fn rfold<Acc, Fold>(self, init: Acc, rfold: Fold) -> Acc
            where
                Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                self.0.rfold(init, rfold)
            }

            fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
                self.0.advance_back_by(n)
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl FusedIterator for $ITER_NAME {}

        #[stable(feature = "exact_size_case_mapping_iter", since = "1.35.0")]
        impl ExactSizeIterator for $ITER_NAME {
            fn len(&self) -> usize {
                self.0.len()
            }

            fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
        }

        // SAFETY: forwards to inner `array::IntoIter`
        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl TrustedLen for $ITER_NAME {}

        // SAFETY: forwards to inner `array::IntoIter`
        #[doc(hidden)]
        #[unstable(feature = "std_internals", issue = "none")]
        unsafe impl TrustedRandomAccessNoCoerce for $ITER_NAME {
            const MAY_HAVE_SIDE_EFFECT: bool = false;
        }

        // SAFETY: this iter has no subtypes/supertypes
        #[doc(hidden)]
        #[unstable(feature = "std_internals", issue = "none")]
        unsafe impl TrustedRandomAccess for $ITER_NAME {}

        #[stable(feature = "char_struct_display", since = "1.16.0")]
        impl fmt::Display for $ITER_NAME {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    }
}

casemappingiter_impls! {
    /// Returns an iterator that yields the lowercase equivalent of a `char`.
    ///
    /// This `struct` is created by the [`to_lowercase`] method on [`char`]. See
    /// its documentation for more.
    ///
    /// [`to_lowercase`]: char::to_lowercase
    ToLowercase
}

casemappingiter_impls! {
    /// Returns an iterator that yields the uppercase equivalent of a `char`.
    ///
    /// This `struct` is created by the [`to_uppercase`] method on [`char`]. See
    /// its documentation for more.
    ///
    /// [`to_uppercase`]: char::to_uppercase
    ToUppercase
}

#[derive(Debug, Clone)]
struct CaseMappingIter(core::array::IntoIter<char, 3>);

impl CaseMappingIter {
    #[inline]
    fn new(chars: [char; 3]) -> CaseMappingIter {
        let mut iter = chars.into_iter();
        if chars[2] == '\0' {
            iter.next_back();
            if chars[1] == '\0' {
                iter.next_back();

                // Deliberately don't check `chars[0]`,
                // as '\0' lowercases to itself
            }
        }
        CaseMappingIter(iter)
    }
}

impl Iterator for CaseMappingIter {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.0.fold(init, fold)
    }

    fn count(self) -> usize {
        self.0.count()
    }

    fn last(self) -> Option<Self::Item> {
        self.0.last()
    }

    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.0.advance_by(n)
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: just forwarding requirements to caller
        unsafe { self.0.__iterator_get_unchecked(idx) }
    }
}

impl DoubleEndedIterator for CaseMappingIter {
    fn next_back(&mut self) -> Option<char> {
        self.0.next_back()
    }

    fn rfold<Acc, Fold>(self, init: Acc, rfold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.0.rfold(init, rfold)
    }

    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.0.advance_back_by(n)
    }
}

impl ExactSizeIterator for CaseMappingIter {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FusedIterator for CaseMappingIter {}

// SAFETY: forwards to inner `array::IntoIter`
unsafe impl TrustedLen for CaseMappingIter {}

// SAFETY: forwards to inner `array::IntoIter`
unsafe impl TrustedRandomAccessNoCoerce for CaseMappingIter {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

// SAFETY: `CaseMappingIter` has no subtypes/supertypes
unsafe impl TrustedRandomAccess for CaseMappingIter {}

impl fmt::Display for CaseMappingIter {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in self.0.clone() {
            f.write_char(c)?;
        }
        Ok(())
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
