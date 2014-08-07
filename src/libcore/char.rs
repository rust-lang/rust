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

#![allow(non_snake_case_functions)]
#![doc(primitive = "char")]

use mem::transmute;
use option::{None, Option, Some};
use iter::range_step;

#[cfg(stage0)]
use iter::Iterator; // NOTE(stage0): Remove after snapshot.

// UTF-8 ranges and tags for encoding characters
static TAG_CONT: u8    = 0b1000_0000u8;
static TAG_TWO_B: u8   = 0b1100_0000u8;
static TAG_THREE_B: u8 = 0b1110_0000u8;
static TAG_FOUR_B: u8  = 0b1111_0000u8;
static MAX_ONE_B: u32   =     0x80u32;
static MAX_TWO_B: u32   =    0x800u32;
static MAX_THREE_B: u32 =  0x10000u32;
static MAX_FOUR_B:  u32 = 0x200000u32;

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
pub static MAX: char = '\U0010ffff';

/// Converts from `u32` to a `char`
#[inline]
pub fn from_u32(i: u32) -> Option<char> {
    // catch out-of-bounds and surrogates
    if (i > MAX as u32) || (i >= 0xD800 && i <= 0xDFFF) {
        None
    } else {
        Some(unsafe { transmute(i) })
    }
}

///
/// Checks if a `char` parses as a numeric digit in the given radix
///
/// Compared to `is_digit()`, this function only recognizes the
/// characters `0-9`, `a-z` and `A-Z`.
///
/// # Return value
///
/// Returns `true` if `c` is a valid digit under `radix`, and `false`
/// otherwise.
///
/// # Failure
///
/// Fails if given a `radix` > 36.
///
/// # Note
///
/// This just wraps `to_digit()`.
///
#[inline]
pub fn is_digit_radix(c: char, radix: uint) -> bool {
    match to_digit(c, radix) {
        Some(_) => true,
        None    => false,
    }
}

///
/// Converts a `char` to the corresponding digit
///
/// # Return value
///
/// If `c` is between '0' and '9', the corresponding value
/// between 0 and 9. If `c` is 'a' or 'A', 10. If `c` is
/// 'b' or 'B', 11, etc. Returns none if the `char` does not
/// refer to a digit in the given radix.
///
/// # Failure
///
/// Fails if given a `radix` outside the range `[0..36]`.
///
#[inline]
pub fn to_digit(c: char, radix: uint) -> Option<uint> {
    if radix > 36 {
        fail!("to_digit: radix is too high (maximum 36)");
    }
    let val = match c {
      '0' .. '9' => c as uint - ('0' as uint),
      'a' .. 'z' => c as uint + 10u - ('a' as uint),
      'A' .. 'Z' => c as uint + 10u - ('A' as uint),
      _ => return None,
    };
    if val < radix { Some(val) }
    else { None }
}

///
/// Converts a number to the character representing it
///
/// # Return value
///
/// Returns `Some(char)` if `num` represents one digit under `radix`,
/// using one character of `0-9` or `a-z`, or `None` if it doesn't.
///
/// # Failure
///
/// Fails if given an `radix` > 36.
///
#[inline]
pub fn from_digit(num: uint, radix: uint) -> Option<char> {
    if radix > 36 {
        fail!("from_digit: radix is to high (maximum 36)");
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

///
/// Returns the hexadecimal Unicode escape of a `char`
///
/// The rules are as follows:
///
/// - chars in [0,0xff] get 2-digit escapes: `\\xNN`
/// - chars in [0x100,0xffff] get 4-digit escapes: `\\uNNNN`
/// - chars above 0x10000 get 8-digit escapes: `\\UNNNNNNNN`
///
pub fn escape_unicode(c: char, f: |char|) {
    // avoid calling str::to_str_radix because we don't really need to allocate
    // here.
    f('\\');
    let pad = match () {
        _ if c <= '\xff'    => { f('x'); 2 }
        _ if c <= '\uffff'  => { f('u'); 4 }
        _                   => { f('U'); 8 }
    };
    for offset in range_step::<i32>(4 * (pad - 1), -1, -4) {
        let offset = offset as uint;
        unsafe {
            match ((c as i32) >> offset) & 0xf {
                i @ 0 .. 9 => { f(transmute('0' as i32 + i)); }
                i => { f(transmute('a' as i32 + (i - 10))); }
            }
        }
    }
}

///
/// Returns a 'default' ASCII and C++11-like literal escape of a `char`
///
/// The default is chosen with a bias toward producing literals that are
/// legal in a variety of languages, including C++11 and similar C-family
/// languages. The exact rules are:
///
/// - Tab, CR and LF are escaped as '\t', '\r' and '\n' respectively.
/// - Single-quote, double-quote and backslash chars are backslash-escaped.
/// - Any other chars in the range [0x20,0x7e] are not escaped.
/// - Any other chars are given hex unicode escapes; see `escape_unicode`.
///
pub fn escape_default(c: char, f: |char|) {
    match c {
        '\t' => { f('\\'); f('t'); }
        '\r' => { f('\\'); f('r'); }
        '\n' => { f('\\'); f('n'); }
        '\\' => { f('\\'); f('\\'); }
        '\'' => { f('\\'); f('\''); }
        '"'  => { f('\\'); f('"'); }
        '\x20' .. '\x7e' => { f(c); }
        _ => c.escape_unicode(f),
    }
}

/// Returns the amount of bytes this `char` would need if encoded in UTF-8
pub fn len_utf8_bytes(c: char) -> uint {
    let code = c as u32;
    match () {
        _ if code < MAX_ONE_B   => 1u,
        _ if code < MAX_TWO_B   => 2u,
        _ if code < MAX_THREE_B => 3u,
        _ if code < MAX_FOUR_B  => 4u,
        _                       => fail!("invalid character!"),
    }
}

/// Basic `char` manipulations.
pub trait Char {
    /// Checks if a `char` parses as a numeric digit in the given radix.
    ///
    /// Compared to `is_digit()`, this function only recognizes the characters
    /// `0-9`, `a-z` and `A-Z`.
    ///
    /// # Return value
    ///
    /// Returns `true` if `c` is a valid digit under `radix`, and `false`
    /// otherwise.
    ///
    /// # Failure
    ///
    /// Fails if given a radix > 36.
    fn is_digit_radix(&self, radix: uint) -> bool;

    /// Converts a character to the corresponding digit.
    ///
    /// # Return value
    ///
    /// If `c` is between '0' and '9', the corresponding value between 0 and
    /// 9. If `c` is 'a' or 'A', 10. If `c` is 'b' or 'B', 11, etc. Returns
    /// none if the character does not refer to a digit in the given radix.
    ///
    /// # Failure
    ///
    /// Fails if given a radix outside the range [0..36].
    fn to_digit(&self, radix: uint) -> Option<uint>;

    /// Converts a number to the character representing it.
    ///
    /// # Return value
    ///
    /// Returns `Some(char)` if `num` represents one digit under `radix`,
    /// using one character of `0-9` or `a-z`, or `None` if it doesn't.
    ///
    /// # Failure
    ///
    /// Fails if given a radix > 36.
    fn from_digit(num: uint, radix: uint) -> Option<Self>;

    /// Returns the hexadecimal Unicode escape of a character.
    ///
    /// The rules are as follows:
    ///
    /// * Characters in [0,0xff] get 2-digit escapes: `\\xNN`
    /// * Characters in [0x100,0xffff] get 4-digit escapes: `\\uNNNN`.
    /// * Characters above 0x10000 get 8-digit escapes: `\\UNNNNNNNN`.
    fn escape_unicode(&self, f: |char|);

    /// Returns a 'default' ASCII and C++11-like literal escape of a
    /// character.
    ///
    /// The default is chosen with a bias toward producing literals that are
    /// legal in a variety of languages, including C++11 and similar C-family
    /// languages. The exact rules are:
    ///
    /// * Tab, CR and LF are escaped as '\t', '\r' and '\n' respectively.
    /// * Single-quote, double-quote and backslash chars are backslash-
    ///   escaped.
    /// * Any other chars in the range [0x20,0x7e] are not escaped.
    /// * Any other chars are given hex unicode escapes; see `escape_unicode`.
    fn escape_default(&self, f: |char|);

    /// Returns the amount of bytes this character would need if encoded in
    /// UTF-8.
    fn len_utf8_bytes(&self) -> uint;

    /// Encodes this character as UTF-8 into the provided byte buffer.
    ///
    /// The buffer must be at least 4 bytes long or a runtime failure may
    /// occur.
    ///
    /// This will then return the number of bytes written to the slice.
    fn encode_utf8(&self, dst: &mut [u8]) -> uint;

    /// Encodes this character as UTF-16 into the provided `u16` buffer.
    ///
    /// The buffer must be at least 2 elements long or a runtime failure may
    /// occur.
    ///
    /// This will then return the number of `u16`s written to the slice.
    fn encode_utf16(&self, dst: &mut [u16]) -> uint;
}

impl Char for char {
    fn is_digit_radix(&self, radix: uint) -> bool { is_digit_radix(*self, radix) }

    fn to_digit(&self, radix: uint) -> Option<uint> { to_digit(*self, radix) }

    fn from_digit(num: uint, radix: uint) -> Option<char> { from_digit(num, radix) }

    fn escape_unicode(&self, f: |char|) { escape_unicode(*self, f) }

    fn escape_default(&self, f: |char|) { escape_default(*self, f) }

    fn len_utf8_bytes(&self) -> uint { len_utf8_bytes(*self) }

    fn encode_utf8<'a>(&self, dst: &'a mut [u8]) -> uint {
        let code = *self as u32;
        if code < MAX_ONE_B {
            dst[0] = code as u8;
            1
        } else if code < MAX_TWO_B {
            dst[0] = (code >> 6u & 0x1F_u32) as u8 | TAG_TWO_B;
            dst[1] = (code & 0x3F_u32) as u8 | TAG_CONT;
            2
        } else if code < MAX_THREE_B {
            dst[0] = (code >> 12u & 0x0F_u32) as u8 | TAG_THREE_B;
            dst[1] = (code >>  6u & 0x3F_u32) as u8 | TAG_CONT;
            dst[2] = (code & 0x3F_u32) as u8 | TAG_CONT;
            3
        } else {
            dst[0] = (code >> 18u & 0x07_u32) as u8 | TAG_FOUR_B;
            dst[1] = (code >> 12u & 0x3F_u32) as u8 | TAG_CONT;
            dst[2] = (code >>  6u & 0x3F_u32) as u8 | TAG_CONT;
            dst[3] = (code & 0x3F_u32) as u8 | TAG_CONT;
            4
        }
    }

    fn encode_utf16(&self, dst: &mut [u16]) -> uint {
        let mut ch = *self as u32;
        if (ch & 0xFFFF_u32) == ch {
            // The BMP falls through (assuming non-surrogate, as it should)
            assert!(ch <= 0xD7FF_u32 || ch >= 0xE000_u32);
            dst[0] = ch as u16;
            1
        } else {
            // Supplementary planes break into surrogates.
            assert!(ch >= 0x1_0000_u32 && ch <= 0x10_FFFF_u32);
            ch -= 0x1_0000_u32;
            dst[0] = 0xD800_u16 | ((ch >> 10) as u16);
            dst[1] = 0xDC00_u16 | ((ch as u16) & 0x3FF_u16);
            2
        }
    }
}
