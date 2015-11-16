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
//! For more details, see ::rustc_unicode::char (a.k.a. std::char)

#![allow(non_snake_case)]
#![stable(feature = "core_char", since = "1.2.0")]

use iter::Iterator;
use mem::transmute;
use option::Option::{None, Some};
use option::Option;
use slice::SliceExt;

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

/// The highest valid code point
#[stable(feature = "rust1", since = "1.0.0")]
pub const MAX: char = '\u{10ffff}';

/// Converts a `u32` to an `Option<char>`.
///
/// # Examples
///
/// ```
/// use std::char;
///
/// assert_eq!(char::from_u32(0x2764), Some('❤'));
/// assert_eq!(char::from_u32(0x110000), None); // invalid character
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_u32(i: u32) -> Option<char> {
    // catch out-of-bounds and surrogates
    if (i > MAX as u32) || (i >= 0xD800 && i <= 0xDFFF) {
        None
    } else {
        Some(unsafe { from_u32_unchecked(i) })
    }
}

/// Converts a `u32` to an `char`, not checking whether it is a valid unicode
/// codepoint.
#[inline]
#[stable(feature = "char_from_unchecked", since = "1.5.0")]
pub unsafe fn from_u32_unchecked(i: u32) -> char {
    transmute(i)
}

/// Converts a number to the character representing it.
///
/// # Return value
///
/// Returns `Some(char)` if `num` represents one digit under `radix`,
/// using one character of `0-9` or `a-z`, or `None` if it doesn't.
///
/// # Panics
///
/// Panics if given an `radix` > 36.
///
/// # Examples
///
/// ```
/// use std::char;
///
/// let c = char::from_digit(4, 10);
///
/// assert_eq!(c, Some('4'));
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
           issue = "27701")]
pub trait CharExt {
    fn is_digit(self, radix: u32) -> bool;
    fn to_digit(self, radix: u32) -> Option<u32>;
    fn escape_unicode(self) -> EscapeUnicode;
    fn escape_default(self) -> EscapeDefault;
    fn len_utf8(self) -> usize;
    fn len_utf16(self) -> usize;
    fn encode_utf8(self, dst: &mut [u8]) -> Option<usize>;
    fn encode_utf16(self, dst: &mut [u16]) -> Option<usize>;
}

#[unstable(feature = "core_char_ext",
           reason = "the stable interface is `impl char` in later crate",
           issue = "27701")]
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
        EscapeUnicode { c: self, state: EscapeUnicodeState::Backslash }
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
    fn encode_utf8(self, dst: &mut [u8]) -> Option<usize> {
        encode_utf8_raw(self as u32, dst)
    }

    #[inline]
    fn encode_utf16(self, dst: &mut [u16]) -> Option<usize> {
        encode_utf16_raw(self as u32, dst)
    }
}

/// Encodes a raw u32 value as UTF-8 into the provided byte buffer,
/// and then returns the number of bytes written.
///
/// If the buffer is not large enough, nothing will be written into it
/// and a `None` will be returned.
#[inline]
#[unstable(feature = "char_internals",
           reason = "this function should not be exposed publicly",
           issue = "0")]
#[doc(hidden)]
pub fn encode_utf8_raw(code: u32, dst: &mut [u8]) -> Option<usize> {
    // Marked #[inline] to allow llvm optimizing it away
    if code < MAX_ONE_B && !dst.is_empty() {
        dst[0] = code as u8;
        Some(1)
    } else if code < MAX_TWO_B && dst.len() >= 2 {
        dst[0] = (code >> 6 & 0x1F) as u8 | TAG_TWO_B;
        dst[1] = (code & 0x3F) as u8 | TAG_CONT;
        Some(2)
    } else if code < MAX_THREE_B && dst.len() >= 3  {
        dst[0] = (code >> 12 & 0x0F) as u8 | TAG_THREE_B;
        dst[1] = (code >>  6 & 0x3F) as u8 | TAG_CONT;
        dst[2] = (code & 0x3F) as u8 | TAG_CONT;
        Some(3)
    } else if dst.len() >= 4 {
        dst[0] = (code >> 18 & 0x07) as u8 | TAG_FOUR_B;
        dst[1] = (code >> 12 & 0x3F) as u8 | TAG_CONT;
        dst[2] = (code >>  6 & 0x3F) as u8 | TAG_CONT;
        dst[3] = (code & 0x3F) as u8 | TAG_CONT;
        Some(4)
    } else {
        None
    }
}

/// Encodes a raw u32 value as UTF-16 into the provided `u16` buffer,
/// and then returns the number of `u16`s written.
///
/// If the buffer is not large enough, nothing will be written into it
/// and a `None` will be returned.
#[inline]
#[unstable(feature = "char_internals",
           reason = "this function should not be exposed publicly",
           issue = "0")]
#[doc(hidden)]
pub fn encode_utf16_raw(mut ch: u32, dst: &mut [u16]) -> Option<usize> {
    // Marked #[inline] to allow llvm optimizing it away
    if (ch & 0xFFFF) == ch && !dst.is_empty() {
        // The BMP falls through (assuming non-surrogate, as it should)
        dst[0] = ch as u16;
        Some(1)
    } else if dst.len() >= 2 {
        // Supplementary planes break into surrogates.
        ch -= 0x1_0000;
        dst[0] = 0xD800 | ((ch >> 10) as u16);
        dst[1] = 0xDC00 | ((ch as u16) & 0x3FF);
        Some(2)
    } else {
        None
    }
}

/// An iterator over the characters that represent a `char`, as escaped by
/// Rust's unicode escaping rules.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeUnicode {
    c: char,
    state: EscapeUnicodeState
}

#[derive(Clone)]
enum EscapeUnicodeState {
    Backslash,
    Type,
    LeftBrace,
    Value(usize),
    RightBrace,
    Done,
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
                let mut n = 0;
                while (self.c as u32) >> (4 * (n + 1)) != 0 {
                    n += 1;
                }
                self.state = EscapeUnicodeState::Value(n);
                Some('{')
            }
            EscapeUnicodeState::Value(offset) => {
                let c = from_digit(((self.c as u32) >> (offset * 4)) & 0xf, 16).unwrap();
                if offset == 0 {
                    self.state = EscapeUnicodeState::RightBrace;
                } else {
                    self.state = EscapeUnicodeState::Value(offset - 1);
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut n = 0;
        while (self.c as usize) >> (4 * (n + 1)) != 0 {
            n += 1;
        }
        let n = match self.state {
            EscapeUnicodeState::Backslash => n + 5,
            EscapeUnicodeState::Type => n + 4,
            EscapeUnicodeState::LeftBrace => n + 3,
            EscapeUnicodeState::Value(offset) => offset + 2,
            EscapeUnicodeState::RightBrace => 1,
            EscapeUnicodeState::Done => 0,
        };
        (n, Some(n))
    }
}

/// An iterator over the characters that represent a `char`, escaped
/// for maximum portability.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeDefault {
    state: EscapeDefaultState
}

#[derive(Clone)]
enum EscapeDefaultState {
    Backslash(char),
    Char(char),
    Done,
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.state {
            EscapeDefaultState::Char(_) => (1, Some(1)),
            EscapeDefaultState::Backslash(_) => (2, Some(2)),
            EscapeDefaultState::Unicode(ref iter) => iter.size_hint(),
            EscapeDefaultState::Done => (0, Some(0)),
        }
    }
}
