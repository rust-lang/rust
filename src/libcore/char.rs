// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Character manipulation.
//!
//! For more details, see ::std_unicode::char (a.k.a. std::char)

#![allow(non_snake_case)]
#![stable(feature = "core_char", since = "1.2.0")]

use char_private::is_printable;
use convert::TryFrom;
use fmt::{self, Write};
use slice;
use iter::FusedIterator;
use mem::transmute;

// UTF-8 ranges and tags for encoding characters
const TAG_CONT: u8    = 0b1000_0000;
const TAG_TWO_B: u8   = 0b1100_0000;
const TAG_THREE_B: u8 = 0b1110_0000;
const TAG_FOUR_B: u8  = 0b1111_0000;
const MAX_ONE_B: u32   =     0x80;
const MAX_TWO_B: u32   =    0x800;
const MAX_THREE_B: u32 =  0x10000;

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

/// The highest valid code point a `char` can have.
///
/// A [`char`] is a [Unicode Scalar Value], which means that it is a [Code
/// Point], but only ones within a certain range. `MAX` is the highest valid
/// code point that's a valid [Unicode Scalar Value].
///
/// [`char`]: ../../std/primitive.char.html
/// [Unicode Scalar Value]: http://www.unicode.org/glossary/#unicode_scalar_value
/// [Code Point]: http://www.unicode.org/glossary/#code_point
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: char = '\u{10ffff}';

/// Converts a `u32` to a `char`.
///
/// Note that all [`char`]s are valid [`u32`]s, and can be casted to one with
/// [`as`]:
///
/// ```
/// let c = '💯';
/// let i = c as u32;
///
/// assert_eq!(128175, i);
/// ```
///
/// However, the reverse is not true: not all valid [`u32`]s are valid
/// [`char`]s. `from_u32()` will return `None` if the input is not a valid value
/// for a [`char`].
///
/// [`char`]: ../../std/primitive.char.html
/// [`u32`]: ../../std/primitive.u32.html
/// [`as`]: ../../book/casting-between-types.html#as
///
/// For an unsafe version of this function which ignores these checks, see
/// [`from_u32_unchecked`].
///
/// [`from_u32_unchecked`]: fn.from_u32_unchecked.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::char;
///
/// let c = char::from_u32(0x2764);
///
/// assert_eq!(Some('❤'), c);
/// ```
///
/// Returning `None` when the input is not a valid [`char`]:
///
/// ```
/// use std::char;
///
/// let c = char::from_u32(0x110000);
///
/// assert_eq!(None, c);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_u32(i: u32) -> Option<char> {
    char::try_from(i).ok()
}

/// Converts a `u32` to a `char`, ignoring validity.
///
/// Note that all [`char`]s are valid [`u32`]s, and can be casted to one with
/// [`as`]:
///
/// ```
/// let c = '💯';
/// let i = c as u32;
///
/// assert_eq!(128175, i);
/// ```
///
/// However, the reverse is not true: not all valid [`u32`]s are valid
/// [`char`]s. `from_u32_unchecked()` will ignore this, and blindly cast to
/// [`char`], possibly creating an invalid one.
///
/// [`char`]: ../../std/primitive.char.html
/// [`u32`]: ../../std/primitive.u32.html
/// [`as`]: ../../book/casting-between-types.html#as
///
/// # Safety
///
/// This function is unsafe, as it may construct invalid `char` values.
///
/// For a safe version of this function, see the [`from_u32`] function.
///
/// [`from_u32`]: fn.from_u32.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::char;
///
/// let c = unsafe { char::from_u32_unchecked(0x2764) };
///
/// assert_eq!('❤', c);
/// ```
#[inline]
#[stable(feature = "char_from_unchecked", since = "1.5.0")]
pub unsafe fn from_u32_unchecked(i: u32) -> char {
    transmute(i)
}

#[stable(feature = "char_convert", since = "1.13.0")]
impl From<char> for u32 {
    #[inline]
    fn from(c: char) -> Self {
        c as u32
    }
}

/// Maps a byte in 0x00...0xFF to a `char` whose code point has the same value, in U+0000 to U+00FF.
///
/// Unicode is designed such that this effectively decodes bytes
/// with the character encoding that IANA calls ISO-8859-1.
/// This encoding is compatible with ASCII.
///
/// Note that this is different from ISO/IEC 8859-1 a.k.a. ISO 8859-1 (with one less hypen),
/// which leaves some "blanks", byte values that are not assigned to any character.
/// ISO-8859-1 (the IANA one) assigns them to the C0 and C1 control codes.
///
/// Note that this is *also* different from Windows-1252 a.k.a. code page 1252,
/// which is a superset ISO/IEC 8859-1 that assigns some (not all!) blanks
/// to punctuation and various Latin characters.
///
/// To confuse things further, [on the Web](https://encoding.spec.whatwg.org/)
/// `ascii`, `iso-8859-1`, and `windows-1252` are all aliases
/// for a superset of Windows-1252 that fills the remaining blanks with corresponding
/// C0 and C1 control codes.
#[stable(feature = "char_convert", since = "1.13.0")]
impl From<u8> for char {
    #[inline]
    fn from(i: u8) -> Self {
        i as char
    }
}

#[unstable(feature = "try_from", issue = "33417")]
impl TryFrom<u32> for char {
    type Err = CharTryFromError;

    #[inline]
    fn try_from(i: u32) -> Result<Self, Self::Err> {
        if (i > MAX as u32) || (i >= 0xD800 && i <= 0xDFFF) {
            Err(CharTryFromError(()))
        } else {
            Ok(unsafe { from_u32_unchecked(i) })
        }
    }
}

/// The error type returned when a conversion from u32 to char fails.
#[unstable(feature = "try_from", issue = "33417")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CharTryFromError(());

#[unstable(feature = "try_from", issue = "33417")]
impl fmt::Display for CharTryFromError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "converted integer out of range for `char`".fmt(f)
    }
}

/// Converts a digit in the given radix to a `char`.
///
/// A 'radix' here is sometimes also called a 'base'. A radix of two
/// indicates a binary number, a radix of ten, decimal, and a radix of
/// sixteen, hexadecimal, to give some common values. Arbitrary
/// radices are supported.
///
/// `from_digit()` will return `None` if the input is not a digit in
/// the given radix.
///
/// # Panics
///
/// Panics if given a radix larger than 36.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::char;
///
/// let c = char::from_digit(4, 10);
///
/// assert_eq!(Some('4'), c);
///
/// // Decimal 11 is a single digit in base 16
/// let c = char::from_digit(11, 16);
///
/// assert_eq!(Some('b'), c);
/// ```
///
/// Returning `None` when the input is not a digit:
///
/// ```
/// use std::char;
///
/// let c = char::from_digit(20, 10);
///
/// assert_eq!(None, c);
/// ```
///
/// Passing a large radix, causing a panic:
///
/// ```
/// use std::thread;
/// use std::char;
///
/// let result = thread::spawn(|| {
///     // this panics
///     let c = char::from_digit(1, 37);
/// }).join();
///
/// assert!(result.is_err());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_digit(num: u32, radix: u32) -> Option<char> {
    if radix > 36 {
        panic!("from_digit: radix is too high (maximum 36)");
    }
    if num < radix {
        let num = num as u8;
        if num < 10 {
            Some((b'0' + num) as char)
        } else {
            Some((b'a' + num - 10) as char)
        }
    } else {
        None
    }
}

// NB: the stabilization and documentation for this trait is in
// unicode/char.rs, not here
#[allow(missing_docs)] // docs in libunicode/u_char.rs
#[doc(hidden)]
#[unstable(feature = "core_char_ext",
           reason = "the stable interface is `impl char` in later crate",
           issue = "32110")]
pub trait CharExt {
    #[stable(feature = "core", since = "1.6.0")]
    fn is_digit(self, radix: u32) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn to_digit(self, radix: u32) -> Option<u32>;
    #[stable(feature = "core", since = "1.6.0")]
    fn escape_unicode(self) -> EscapeUnicode;
    #[stable(feature = "core", since = "1.6.0")]
    fn escape_default(self) -> EscapeDefault;
    #[unstable(feature = "char_escape_debug", issue = "35068")]
    fn escape_debug(self) -> EscapeDebug;
    #[stable(feature = "core", since = "1.6.0")]
    fn len_utf8(self) -> usize;
    #[stable(feature = "core", since = "1.6.0")]
    fn len_utf16(self) -> usize;
    #[stable(feature = "unicode_encode_char", since = "1.15.0")]
    fn encode_utf8(self, dst: &mut [u8]) -> &mut str;
    #[stable(feature = "unicode_encode_char", since = "1.15.0")]
    fn encode_utf16(self, dst: &mut [u16]) -> &mut [u16];
}

#[stable(feature = "core", since = "1.6.0")]
impl CharExt for char {
    #[inline]
    fn is_digit(self, radix: u32) -> bool {
        self.to_digit(radix).is_some()
    }

    #[inline]
    fn to_digit(self, radix: u32) -> Option<u32> {
        if radix > 36 {
            panic!("to_digit: radix is too high (maximum 36)");
        }
        let val = match self {
          '0' ... '9' => self as u32 - '0' as u32,
          'a' ... 'z' => self as u32 - 'a' as u32 + 10,
          'A' ... 'Z' => self as u32 - 'A' as u32 + 10,
          _ => return None,
        };
        if val < radix { Some(val) }
        else { None }
    }

    #[inline]
    fn escape_unicode(self) -> EscapeUnicode {
        let c = self as u32;

        // or-ing 1 ensures that for c==0 the code computes that one
        // digit should be printed and (which is the same) avoids the
        // (31 - 32) underflow
        let msb = 31 - (c | 1).leading_zeros();

        // the index of the most significant hex digit
        let ms_hex_digit = msb / 4;
        EscapeUnicode {
            c: self,
            state: EscapeUnicodeState::Backslash,
            hex_digit_idx: ms_hex_digit as usize,
        }
    }

    #[inline]
    fn escape_default(self) -> EscapeDefault {
        let init_state = match self {
            '\t' => EscapeDefaultState::Backslash('t'),
            '\r' => EscapeDefaultState::Backslash('r'),
            '\n' => EscapeDefaultState::Backslash('n'),
            '\\' | '\'' | '"' => EscapeDefaultState::Backslash(self),
            '\x20' ... '\x7e' => EscapeDefaultState::Char(self),
            _ => EscapeDefaultState::Unicode(self.escape_unicode())
        };
        EscapeDefault { state: init_state }
    }

    #[inline]
    fn escape_debug(self) -> EscapeDebug {
        let init_state = match self {
            '\t' => EscapeDefaultState::Backslash('t'),
            '\r' => EscapeDefaultState::Backslash('r'),
            '\n' => EscapeDefaultState::Backslash('n'),
            '\\' | '\'' | '"' => EscapeDefaultState::Backslash(self),
            c if is_printable(c) => EscapeDefaultState::Char(c),
            c => EscapeDefaultState::Unicode(c.escape_unicode()),
        };
        EscapeDebug(EscapeDefault { state: init_state })
    }

    #[inline]
    fn len_utf8(self) -> usize {
        let code = self as u32;
        if code < MAX_ONE_B {
            1
        } else if code < MAX_TWO_B {
            2
        } else if code < MAX_THREE_B {
            3
        } else {
            4
        }
    }

    #[inline]
    fn len_utf16(self) -> usize {
        let ch = self as u32;
        if (ch & 0xFFFF) == ch { 1 } else { 2 }
    }

    #[inline]
    fn encode_utf8(self, dst: &mut [u8]) -> &mut str {
        let code = self as u32;
        unsafe {
            let len =
            if code < MAX_ONE_B && !dst.is_empty() {
                *dst.get_unchecked_mut(0) = code as u8;
                1
            } else if code < MAX_TWO_B && dst.len() >= 2 {
                *dst.get_unchecked_mut(0) = (code >> 6 & 0x1F) as u8 | TAG_TWO_B;
                *dst.get_unchecked_mut(1) = (code & 0x3F) as u8 | TAG_CONT;
                2
            } else if code < MAX_THREE_B && dst.len() >= 3  {
                *dst.get_unchecked_mut(0) = (code >> 12 & 0x0F) as u8 | TAG_THREE_B;
                *dst.get_unchecked_mut(1) = (code >>  6 & 0x3F) as u8 | TAG_CONT;
                *dst.get_unchecked_mut(2) = (code & 0x3F) as u8 | TAG_CONT;
                3
            } else if dst.len() >= 4 {
                *dst.get_unchecked_mut(0) = (code >> 18 & 0x07) as u8 | TAG_FOUR_B;
                *dst.get_unchecked_mut(1) = (code >> 12 & 0x3F) as u8 | TAG_CONT;
                *dst.get_unchecked_mut(2) = (code >>  6 & 0x3F) as u8 | TAG_CONT;
                *dst.get_unchecked_mut(3) = (code & 0x3F) as u8 | TAG_CONT;
                4
            } else {
                panic!("encode_utf8: need {} bytes to encode U+{:X}, but the buffer has {}",
                    from_u32_unchecked(code).len_utf8(),
                    code,
                    dst.len())
            };
            transmute(slice::from_raw_parts_mut(dst.as_mut_ptr(), len))
        }
    }

    #[inline]
    fn encode_utf16(self, dst: &mut [u16]) -> &mut [u16] {
        let mut code = self as u32;
        unsafe {
            if (code & 0xFFFF) == code && !dst.is_empty() {
                // The BMP falls through (assuming non-surrogate, as it should)
                *dst.get_unchecked_mut(0) = code as u16;
                slice::from_raw_parts_mut(dst.as_mut_ptr(), 1)
            } else if dst.len() >= 2 {
                // Supplementary planes break into surrogates.
                code -= 0x1_0000;
                *dst.get_unchecked_mut(0) = 0xD800 | ((code >> 10) as u16);
                *dst.get_unchecked_mut(1) = 0xDC00 | ((code as u16) & 0x3FF);
                slice::from_raw_parts_mut(dst.as_mut_ptr(), 2)
            } else {
                panic!("encode_utf16: need {} units to encode U+{:X}, but the buffer has {}",
                    from_u32_unchecked(code).len_utf16(),
                    code,
                    dst.len())
            }
        }
    }
}

/// Returns an iterator that yields the hexadecimal Unicode escape of a
/// character, as `char`s.
///
/// This `struct` is created by the [`escape_unicode`] method on [`char`]. See
/// its documentation for more.
///
/// [`escape_unicode`]: ../../std/primitive.char.html#method.escape_unicode
/// [`char`]: ../../std/primitive.char.html
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeUnicode {
    c: char,
    state: EscapeUnicodeState,

    // The index of the next hex digit to be printed (0 if none),
    // i.e. the number of remaining hex digits to be printed;
    // increasing from the least significant digit: 0x543210
    hex_digit_idx: usize,
}

// The enum values are ordered so that their representation is the
// same as the remaining length (besides the hexadecimal digits). This
// likely makes `len()` a single load from memory) and inline-worth.
#[derive(Clone, Debug)]
enum EscapeUnicodeState {
    Done,
    RightBrace,
    Value,
    LeftBrace,
    Type,
    Backslash,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EscapeUnicode {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        match self.state {
            EscapeUnicodeState::Backslash => {
                self.state = EscapeUnicodeState::Type;
                Some('\\')
            }
            EscapeUnicodeState::Type => {
                self.state = EscapeUnicodeState::LeftBrace;
                Some('u')
            }
            EscapeUnicodeState::LeftBrace => {
                self.state = EscapeUnicodeState::Value;
                Some('{')
            }
            EscapeUnicodeState::Value => {
                let hex_digit = ((self.c as u32) >> (self.hex_digit_idx * 4)) & 0xf;
                let c = from_digit(hex_digit, 16).unwrap();
                if self.hex_digit_idx == 0 {
                    self.state = EscapeUnicodeState::RightBrace;
                } else {
                    self.hex_digit_idx -= 1;
                }
                Some(c)
            }
            EscapeUnicodeState::RightBrace => {
                self.state = EscapeUnicodeState::Done;
                Some('}')
            }
            EscapeUnicodeState::Done => None,
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

    fn last(self) -> Option<char> {
        match self.state {
            EscapeUnicodeState::Done => None,

            EscapeUnicodeState::RightBrace |
            EscapeUnicodeState::Value |
            EscapeUnicodeState::LeftBrace |
            EscapeUnicodeState::Type |
            EscapeUnicodeState::Backslash => Some('}'),
        }
    }
}

#[stable(feature = "exact_size_escape", since = "1.11.0")]
impl ExactSizeIterator for EscapeUnicode {
    #[inline]
    fn len(&self) -> usize {
        // The match is a single memory access with no branching
        self.hex_digit_idx + match self.state {
            EscapeUnicodeState::Done => 0,
            EscapeUnicodeState::RightBrace => 1,
            EscapeUnicodeState::Value => 2,
            EscapeUnicodeState::LeftBrace => 3,
            EscapeUnicodeState::Type => 4,
            EscapeUnicodeState::Backslash => 5,
        }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl FusedIterator for EscapeUnicode {}

#[stable(feature = "char_struct_display", since = "1.16.0")]
impl fmt::Display for EscapeUnicode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            f.write_char(c)?;
        }
        Ok(())
    }
}

/// An iterator that yields the literal escape code of a `char`.
///
/// This `struct` is created by the [`escape_default`] method on [`char`]. See
/// its documentation for more.
///
/// [`escape_default`]: ../../std/primitive.char.html#method.escape_default
/// [`char`]: ../../std/primitive.char.html
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeDefault {
    state: EscapeDefaultState
}

#[derive(Clone, Debug)]
enum EscapeDefaultState {
    Done,
    Char(char),
    Backslash(char),
    Unicode(EscapeUnicode),
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EscapeDefault {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        match self.state {
            EscapeDefaultState::Backslash(c) => {
                self.state = EscapeDefaultState::Char(c);
                Some('\\')
            }
            EscapeDefaultState::Char(c) => {
                self.state = EscapeDefaultState::Done;
                Some(c)
            }
            EscapeDefaultState::Done => None,
            EscapeDefaultState::Unicode(ref mut iter) => iter.next(),
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

    fn nth(&mut self, n: usize) -> Option<char> {
        match self.state {
            EscapeDefaultState::Backslash(c) if n == 0 => {
                self.state = EscapeDefaultState::Char(c);
                Some('\\')
            },
            EscapeDefaultState::Backslash(c) if n == 1 => {
                self.state = EscapeDefaultState::Done;
                Some(c)
            },
            EscapeDefaultState::Backslash(_) => {
                self.state = EscapeDefaultState::Done;
                None
            },
            EscapeDefaultState::Char(c) => {
                self.state = EscapeDefaultState::Done;

                if n == 0 {
                    Some(c)
                } else {
                    None
                }
            },
            EscapeDefaultState::Done => return None,
            EscapeDefaultState::Unicode(ref mut i) => return i.nth(n),
        }
    }

    fn last(self) -> Option<char> {
        match self.state {
            EscapeDefaultState::Unicode(iter) => iter.last(),
            EscapeDefaultState::Done => None,
            EscapeDefaultState::Backslash(c) | EscapeDefaultState::Char(c) => Some(c),
        }
    }
}

#[stable(feature = "exact_size_escape", since = "1.11.0")]
impl ExactSizeIterator for EscapeDefault {
    fn len(&self) -> usize {
        match self.state {
            EscapeDefaultState::Done => 0,
            EscapeDefaultState::Char(_) => 1,
            EscapeDefaultState::Backslash(_) => 2,
            EscapeDefaultState::Unicode(ref iter) => iter.len(),
        }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl FusedIterator for EscapeDefault {}

#[stable(feature = "char_struct_display", since = "1.16.0")]
impl fmt::Display for EscapeDefault {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            f.write_char(c)?;
        }
        Ok(())
    }
}

/// An iterator that yields the literal escape code of a `char`.
///
/// This `struct` is created by the [`escape_debug`] method on [`char`]. See its
/// documentation for more.
///
/// [`escape_debug`]: ../../std/primitive.char.html#method.escape_debug
/// [`char`]: ../../std/primitive.char.html
#[unstable(feature = "char_escape_debug", issue = "35068")]
#[derive(Clone, Debug)]
pub struct EscapeDebug(EscapeDefault);

#[unstable(feature = "char_escape_debug", issue = "35068")]
impl Iterator for EscapeDebug {
    type Item = char;
    fn next(&mut self) -> Option<char> { self.0.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

#[unstable(feature = "char_escape_debug", issue = "35068")]
impl ExactSizeIterator for EscapeDebug { }

#[unstable(feature = "fused", issue = "35602")]
impl FusedIterator for EscapeDebug {}

#[unstable(feature = "char_escape_debug", issue = "35068")]
impl fmt::Display for EscapeDebug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}



/// An iterator over an iterator of bytes of the characters the bytes represent
/// as UTF-8
#[unstable(feature = "decode_utf8", issue = "33906")]
#[derive(Clone, Debug)]
pub struct DecodeUtf8<I: Iterator<Item = u8>>(::iter::Peekable<I>);

/// Decodes an `Iterator` of bytes as UTF-8.
#[unstable(feature = "decode_utf8", issue = "33906")]
#[inline]
pub fn decode_utf8<I: IntoIterator<Item = u8>>(i: I) -> DecodeUtf8<I::IntoIter> {
    DecodeUtf8(i.into_iter().peekable())
}

/// `<DecodeUtf8 as Iterator>::next` returns this for an invalid input sequence.
#[unstable(feature = "decode_utf8", issue = "33906")]
#[derive(PartialEq, Eq, Debug)]
pub struct InvalidSequence(());

#[unstable(feature = "decode_utf8", issue = "33906")]
impl<I: Iterator<Item = u8>> Iterator for DecodeUtf8<I> {
    type Item = Result<char, InvalidSequence>;
    #[inline]

    fn next(&mut self) -> Option<Result<char, InvalidSequence>> {
        self.0.next().map(|first_byte| {
            // Emit InvalidSequence according to
            // Unicode §5.22 Best Practice for U+FFFD Substitution
            // http://www.unicode.org/versions/Unicode9.0.0/ch05.pdf#G40630

            // Roughly: consume at least one byte,
            // then validate one byte at a time and stop before the first unexpected byte
            // (which might be the valid start of the next byte sequence).

            let mut code_point;
            macro_rules! first_byte {
                ($mask: expr) => {
                    code_point = u32::from(first_byte & $mask)
                }
            }
            macro_rules! continuation_byte {
                () => { continuation_byte!(0x80...0xBF) };
                ($range: pat) => {
                    match self.0.peek() {
                        Some(&byte @ $range) => {
                            code_point = (code_point << 6) | u32::from(byte & 0b0011_1111);
                            self.0.next();
                        }
                        _ => return Err(InvalidSequence(()))
                    }
                }
            }

            match first_byte {
                0x00...0x7F => {
                    first_byte!(0b1111_1111);
                }
                0xC2...0xDF => {
                    first_byte!(0b0001_1111);
                    continuation_byte!();
                }
                0xE0 => {
                    first_byte!(0b0000_1111);
                    continuation_byte!(0xA0...0xBF);  // 0x80...0x9F here are overlong
                    continuation_byte!();
                }
                0xE1...0xEC | 0xEE...0xEF => {
                    first_byte!(0b0000_1111);
                    continuation_byte!();
                    continuation_byte!();
                }
                0xED => {
                    first_byte!(0b0000_1111);
                    continuation_byte!(0x80...0x9F);  // 0xA0..0xBF here are surrogates
                    continuation_byte!();
                }
                0xF0 => {
                    first_byte!(0b0000_0111);
                    continuation_byte!(0x90...0xBF);  // 0x80..0x8F here are overlong
                    continuation_byte!();
                    continuation_byte!();
                }
                0xF1...0xF3 => {
                    first_byte!(0b0000_0111);
                    continuation_byte!();
                    continuation_byte!();
                    continuation_byte!();
                }
                0xF4 => {
                    first_byte!(0b0000_0111);
                    continuation_byte!(0x80...0x8F);  // 0x90..0xBF here are beyond char::MAX
                    continuation_byte!();
                    continuation_byte!();
                }
                _ => return Err(InvalidSequence(()))  // Illegal first byte, overlong, or beyond MAX
            }
            unsafe {
                Ok(from_u32_unchecked(code_point))
            }
        })
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<I: FusedIterator<Item = u8>> FusedIterator for DecodeUtf8<I> {}
