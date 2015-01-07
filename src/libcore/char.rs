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
//! For more details, see ::unicode::char (a.k.a. std::char)

#![allow(non_snake_case)]
#![doc(primitive = "char")]

use iter::Iterator;
use mem::transmute;
use option::Option::{None, Some};
use option::Option;
use slice::SliceExt;

// UTF-8 ranges and tags for encoding characters
static TAG_CONT: u8    = 0b1000_0000u8;
static TAG_TWO_B: u8   = 0b1100_0000u8;
static TAG_THREE_B: u8 = 0b1110_0000u8;
static TAG_FOUR_B: u8  = 0b1111_0000u8;
static MAX_ONE_B: u32   =     0x80u32;
static MAX_TWO_B: u32   =    0x800u32;
static MAX_THREE_B: u32 =  0x10000u32;

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
#[stable]
pub const MAX: char = '\u{10ffff}';

/// Converts from `u32` to a `char`
#[inline]
#[stable]
pub fn from_u32(i: u32) -> Option<char> {
    // catch out-of-bounds and surrogates
    if (i > MAX as u32) || (i >= 0xD800 && i <= 0xDFFF) {
        None
    } else {
        Some(unsafe { transmute(i) })
    }
}

///
/// Converts a number to the character representing it
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
#[inline]
#[unstable = "pending integer conventions"]
pub fn from_digit(num: uint, radix: uint) -> Option<char> {
    if radix > 36 {
        panic!("from_digit: radix is too high (maximum 36)");
    }
    if num < radix {
        unsafe {
            if num < 10 {
                Some(transmute(('0' as uint + num) as u32))
            } else {
                Some(transmute(('a' as uint + num - 10u) as u32))
            }
        }
    } else {
        None
    }
}

/// Basic `char` manipulations.
#[stable]
pub trait CharExt {
    /// Checks if a `char` parses as a numeric digit in the given radix.
    ///
    /// Compared to `is_numeric()`, this function only recognizes the characters
    /// `0-9`, `a-z` and `A-Z`.
    ///
    /// # Return value
    ///
    /// Returns `true` if `c` is a valid digit under `radix`, and `false`
    /// otherwise.
    ///
    /// # Panics
    ///
    /// Panics if given a radix > 36.
    #[unstable = "pending integer conventions"]
    fn is_digit(self, radix: uint) -> bool;

    /// Converts a character to the corresponding digit.
    ///
    /// # Return value
    ///
    /// If `c` is between '0' and '9', the corresponding value between 0 and
    /// 9. If `c` is 'a' or 'A', 10. If `c` is 'b' or 'B', 11, etc. Returns
    /// none if the character does not refer to a digit in the given radix.
    ///
    /// # Panics
    ///
    /// Panics if given a radix outside the range [0..36].
    #[unstable = "pending integer conventions"]
    fn to_digit(self, radix: uint) -> Option<uint>;

    /// Returns an iterator that yields the hexadecimal Unicode escape
    /// of a character, as `char`s.
    ///
    /// All characters are escaped with Rust syntax of the form `\\u{NNNN}`
    /// where `NNNN` is the shortest hexadecimal representation of the code
    /// point.
    #[stable]
    fn escape_unicode(self) -> EscapeUnicode;

    /// Returns an iterator that yields the 'default' ASCII and
    /// C++11-like literal escape of a character, as `char`s.
    ///
    /// The default is chosen with a bias toward producing literals that are
    /// legal in a variety of languages, including C++11 and similar C-family
    /// languages. The exact rules are:
    ///
    /// * Tab, CR and LF are escaped as '\t', '\r' and '\n' respectively.
    /// * Single-quote, double-quote and backslash chars are backslash-
    ///   escaped.
    /// * Any other chars in the range [0x20,0x7e] are not escaped.
    /// * Any other chars are given hex Unicode escapes; see `escape_unicode`.
    #[stable]
    fn escape_default(self) -> EscapeDefault;

    /// Returns the amount of bytes this character would need if encoded in
    /// UTF-8.
    #[stable]
    fn len_utf8(self) -> uint;

    /// Returns the amount of bytes this character would need if encoded in
    /// UTF-16.
    #[stable]
    fn len_utf16(self) -> uint;

    /// Encodes this character as UTF-8 into the provided byte buffer,
    /// and then returns the number of bytes written.
    ///
    /// If the buffer is not large enough, nothing will be written into it
    /// and a `None` will be returned.
    #[stable]
    fn encode_utf8(self, dst: &mut [u8]) -> Option<uint>;

    /// Encodes this character as UTF-16 into the provided `u16` buffer,
    /// and then returns the number of `u16`s written.
    ///
    /// If the buffer is not large enough, nothing will be written into it
    /// and a `None` will be returned.
    #[stable]
    fn encode_utf16(self, dst: &mut [u16]) -> Option<uint>;
}

#[stable]
impl CharExt for char {
    #[unstable = "pending integer conventions"]
    fn is_digit(self, radix: uint) -> bool {
        self.to_digit(radix).is_some()
    }

    #[unstable = "pending integer conventions"]
    fn to_digit(self, radix: uint) -> Option<uint> {
        if radix > 36 {
            panic!("to_digit: radix is too high (maximum 36)");
        }
        let val = match self {
          '0' ... '9' => self as uint - ('0' as uint),
          'a' ... 'z' => self as uint + 10u - ('a' as uint),
          'A' ... 'Z' => self as uint + 10u - ('A' as uint),
          _ => return None,
        };
        if val < radix { Some(val) }
        else { None }
    }

    #[stable]
    fn escape_unicode(self) -> EscapeUnicode {
        EscapeUnicode { c: self, state: EscapeUnicodeState::Backslash }
    }

    #[stable]
    fn escape_default(self) -> EscapeDefault {
        let init_state = match self {
            '\t' => EscapeDefaultState::Backslash('t'),
            '\r' => EscapeDefaultState::Backslash('r'),
            '\n' => EscapeDefaultState::Backslash('n'),
            '\\' => EscapeDefaultState::Backslash('\\'),
            '\'' => EscapeDefaultState::Backslash('\''),
            '"'  => EscapeDefaultState::Backslash('"'),
            '\x20' ... '\x7e' => EscapeDefaultState::Char(self),
            _ => EscapeDefaultState::Unicode(self.escape_unicode())
        };
        EscapeDefault { state: init_state }
    }

    #[inline]
    #[stable]
    fn len_utf8(self) -> uint {
        let code = self as u32;
        match () {
            _ if code < MAX_ONE_B   => 1u,
            _ if code < MAX_TWO_B   => 2u,
            _ if code < MAX_THREE_B => 3u,
            _  => 4u,
        }
    }

    #[inline]
    #[stable]
    fn len_utf16(self) -> uint {
        let ch = self as u32;
        if (ch & 0xFFFF_u32) == ch { 1 } else { 2 }
    }

    #[inline]
    #[unstable = "pending decision about Iterator/Writer/Reader"]
    fn encode_utf8(self, dst: &mut [u8]) -> Option<uint> {
        // Marked #[inline] to allow llvm optimizing it away
        let code = self as u32;
        if code < MAX_ONE_B && dst.len() >= 1 {
            dst[0] = code as u8;
            Some(1)
        } else if code < MAX_TWO_B && dst.len() >= 2 {
            dst[0] = (code >> 6u & 0x1F_u32) as u8 | TAG_TWO_B;
            dst[1] = (code & 0x3F_u32) as u8 | TAG_CONT;
            Some(2)
        } else if code < MAX_THREE_B && dst.len() >= 3  {
            dst[0] = (code >> 12u & 0x0F_u32) as u8 | TAG_THREE_B;
            dst[1] = (code >>  6u & 0x3F_u32) as u8 | TAG_CONT;
            dst[2] = (code & 0x3F_u32) as u8 | TAG_CONT;
            Some(3)
        } else if dst.len() >= 4 {
            dst[0] = (code >> 18u & 0x07_u32) as u8 | TAG_FOUR_B;
            dst[1] = (code >> 12u & 0x3F_u32) as u8 | TAG_CONT;
            dst[2] = (code >>  6u & 0x3F_u32) as u8 | TAG_CONT;
            dst[3] = (code & 0x3F_u32) as u8 | TAG_CONT;
            Some(4)
        } else {
            None
        }
    }

    #[inline]
    #[unstable = "pending decision about Iterator/Writer/Reader"]
    fn encode_utf16(self, dst: &mut [u16]) -> Option<uint> {
        // Marked #[inline] to allow llvm optimizing it away
        let mut ch = self as u32;
        if (ch & 0xFFFF_u32) == ch  && dst.len() >= 1 {
            // The BMP falls through (assuming non-surrogate, as it should)
            dst[0] = ch as u16;
            Some(1)
        } else if dst.len() >= 2 {
            // Supplementary planes break into surrogates.
            ch -= 0x1_0000_u32;
            dst[0] = 0xD800_u16 | ((ch >> 10) as u16);
            dst[1] = 0xDC00_u16 | ((ch as u16) & 0x3FF_u16);
            Some(2)
        } else {
            None
        }
    }
}

/// An iterator over the characters that represent a `char`, as escaped by
/// Rust's unicode escaping rules.
#[derive(Clone)]
#[stable]
pub struct EscapeUnicode {
    c: char,
    state: EscapeUnicodeState
}

#[derive(Clone)]
#[unstable]
enum EscapeUnicodeState {
    Backslash,
    Type,
    LeftBrace,
    Value(uint),
    RightBrace,
    Done,
}

#[stable]
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
                let mut n = 0u;
                while (self.c as u32) >> (4 * (n + 1)) != 0 {
                    n += 1;
                }
                self.state = EscapeUnicodeState::Value(n);
                Some('{')
            }
            EscapeUnicodeState::Value(offset) => {
                let v = match ((self.c as i32) >> (offset * 4)) & 0xf {
                    i @ 0 ... 9 => '0' as i32 + i,
                    i => 'a' as i32 + (i - 10)
                };
                if offset == 0 {
                    self.state = EscapeUnicodeState::RightBrace;
                } else {
                    self.state = EscapeUnicodeState::Value(offset - 1);
                }
                Some(unsafe { transmute(v) })
            }
            EscapeUnicodeState::RightBrace => {
                self.state = EscapeUnicodeState::Done;
                Some('}')
            }
            EscapeUnicodeState::Done => None,
        }
    }
}

/// An iterator over the characters that represent a `char`, escaped
/// for maximum portability.
#[derive(Clone)]
#[stable]
pub struct EscapeDefault {
    state: EscapeDefaultState
}

#[derive(Clone)]
#[unstable]
enum EscapeDefaultState {
    Backslash(char),
    Char(char),
    Done,
    Unicode(EscapeUnicode),
}

#[stable]
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
            EscapeDefaultState::Unicode(ref mut iter) => iter.next()
        }
    }
}
