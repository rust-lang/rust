// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Character manipulation (`char` type, Unicode Scalar Value)
//!
//! This module  provides the `Char` trait, as well as its implementation
//! for the primitive `char` type, in order to allow basic character manipulation.
//!
//! A `char` actually represents a
//! *[Unicode Scalar Value](http://www.unicode.org/glossary/#unicode_scalar_value)*,
//! as it can contain any Unicode code point except high-surrogate and
//! low-surrogate code points.
//!
//! As such, only values in the ranges \[0x0,0xD7FF\] and \[0xE000,0x10FFFF\]
//! (inclusive) are allowed. A `char` can always be safely cast to a `u32`;
//! however the converse is not always true due to the above range limits
//! and, as such, should be performed via the `from_u32` function..


use mem::transmute;
use option::{None, Option, Some};
use iter::{Iterator, range_step};
use unicode::{derived_property, property, general_category, conversions};

/// Returns the canonical decomposition of a character.
pub use unicode::normalization::decompose_canonical;
/// Returns the compatibility decomposition of a character.
pub use unicode::normalization::decompose_compatible;

#[cfg(not(test))] use cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering};
#[cfg(not(test))] use default::Default;

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

/// Returns whether the specified `char` is considered a Unicode alphabetic
/// code point
pub fn is_alphabetic(c: char) -> bool   { derived_property::Alphabetic(c) }

/// Returns whether the specified `char` satisfies the 'XID_Start' Unicode property
///
/// 'XID_Start' is a Unicode Derived Property specified in
/// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
/// mostly similar to ID_Start but modified for closure under NFKx.
pub fn is_XID_start(c: char) -> bool    { derived_property::XID_Start(c) }

/// Returns whether the specified `char` satisfies the 'XID_Continue' Unicode property
///
/// 'XID_Continue' is a Unicode Derived Property specified in
/// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
/// mostly similar to 'ID_Continue' but modified for closure under NFKx.
pub fn is_XID_continue(c: char) -> bool { derived_property::XID_Continue(c) }

///
/// Indicates whether a `char` is in lower case
///
/// This is defined according to the terms of the Unicode Derived Core Property 'Lowercase'.
///
#[inline]
pub fn is_lowercase(c: char) -> bool { derived_property::Lowercase(c) }

///
/// Indicates whether a `char` is in upper case
///
/// This is defined according to the terms of the Unicode Derived Core Property 'Uppercase'.
///
#[inline]
pub fn is_uppercase(c: char) -> bool { derived_property::Uppercase(c) }

///
/// Indicates whether a `char` is whitespace
///
/// Whitespace is defined in terms of the Unicode Property 'White_Space'.
///
#[inline]
pub fn is_whitespace(c: char) -> bool {
    // As an optimization ASCII whitespace characters are checked separately
    c == ' '
        || ('\x09' <= c && c <= '\x0d')
        || property::White_Space(c)
}

///
/// Indicates whether a `char` is alphanumeric
///
/// Alphanumericness is defined in terms of the Unicode General Categories
/// 'Nd', 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
///
#[inline]
pub fn is_alphanumeric(c: char) -> bool {
    derived_property::Alphabetic(c)
        || general_category::Nd(c)
        || general_category::Nl(c)
        || general_category::No(c)
}

///
/// Indicates whether a `char` is a control code point
///
/// Control code points are defined in terms of the Unicode General Category
/// 'Cc'.
///
#[inline]
pub fn is_control(c: char) -> bool { general_category::Cc(c) }

/// Indicates whether the `char` is numeric (Nd, Nl, or No)
#[inline]
pub fn is_digit(c: char) -> bool {
    general_category::Nd(c)
        || general_category::Nl(c)
        || general_category::No(c)
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

/// Convert a char to its uppercase equivalent
///
/// The case-folding performed is the common or simple mapping:
/// it maps one unicode codepoint (one char in Rust) to its uppercase equivalent according
/// to the Unicode database at ftp://ftp.unicode.org/Public/UNIDATA/UnicodeData.txt
/// The additional SpecialCasing.txt is not considered here, as it expands to multiple
/// codepoints in some cases.
///
/// A full reference can be found here
/// http://www.unicode.org/versions/Unicode4.0.0/ch03.pdf#G33992
///
/// # Return value
///
/// Returns the char itself if no conversion was made
#[inline]
pub fn to_uppercase(c: char) -> char {
    conversions::to_upper(c)
}

/// Convert a char to its lowercase equivalent
///
/// The case-folding performed is the common or simple mapping
/// see `to_uppercase` for references and more information
///
/// # Return value
///
/// Returns the char itself if no conversion if possible
#[inline]
pub fn to_lowercase(c: char) -> char {
    conversions::to_lower(c)
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

/// Useful functions for Unicode characters.
pub trait Char {
    /// Returns whether the specified character is considered a Unicode
    /// alphabetic code point.
    fn is_alphabetic(&self) -> bool;

    /// Returns whether the specified character satisfies the 'XID_Start'
    /// Unicode property.
    ///
    /// 'XID_Start' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to ID_Start but modified for closure under NFKx.
    fn is_XID_start(&self) -> bool;

    /// Returns whether the specified `char` satisfies the 'XID_Continue'
    /// Unicode property.
    ///
    /// 'XID_Continue' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to 'ID_Continue' but modified for closure under NFKx.
    fn is_XID_continue(&self) -> bool;


    /// Indicates whether a character is in lowercase.
    ///
    /// This is defined according to the terms of the Unicode Derived Core
    /// Property `Lowercase`.
    fn is_lowercase(&self) -> bool;

    /// Indicates whether a character is in uppercase.
    ///
    /// This is defined according to the terms of the Unicode Derived Core
    /// Property `Uppercase`.
    fn is_uppercase(&self) -> bool;

    /// Indicates whether a character is whitespace.
    ///
    /// Whitespace is defined in terms of the Unicode Property `White_Space`.
    fn is_whitespace(&self) -> bool;

    /// Indicates whether a character is alphanumeric.
    ///
    /// Alphanumericness is defined in terms of the Unicode General Categories
    /// 'Nd', 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
    fn is_alphanumeric(&self) -> bool;

    /// Indicates whether a character is a control code point.
    ///
    /// Control code points are defined in terms of the Unicode General
    /// Category `Cc`.
    fn is_control(&self) -> bool;

    /// Indicates whether the character is numeric (Nd, Nl, or No).
    fn is_digit(&self) -> bool;

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

    /// Converts a character to its lowercase equivalent.
    ///
    /// The case-folding performed is the common or simple mapping. See
    /// `to_uppercase()` for references and more information.
    ///
    /// # Return value
    ///
    /// Returns the lowercase equivalent of the character, or the character
    /// itself if no conversion is possible.
    fn to_lowercase(&self) -> char;

    /// Converts a character to its uppercase equivalent.
    ///
    /// The case-folding performed is the common or simple mapping: it maps
    /// one unicode codepoint (one character in Rust) to its uppercase
    /// equivalent according to the Unicode database [1]. The additional
    /// `SpecialCasing.txt` is not considered here, as it expands to multiple
    /// codepoints in some cases.
    ///
    /// A full reference can be found here [2].
    ///
    /// # Return value
    ///
    /// Returns the uppercase equivalent of the character, or the character
    /// itself if no conversion was made.
    ///
    /// [1]: ftp://ftp.unicode.org/Public/UNIDATA/UnicodeData.txt
    ///
    /// [2]: http://www.unicode.org/versions/Unicode4.0.0/ch03.pdf#G33992
    fn to_uppercase(&self) -> char;

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
    fn from_digit(num: uint, radix: uint) -> Option<char>;

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
    fn is_alphabetic(&self) -> bool { is_alphabetic(*self) }

    fn is_XID_start(&self) -> bool { is_XID_start(*self) }

    fn is_XID_continue(&self) -> bool { is_XID_continue(*self) }

    fn is_lowercase(&self) -> bool { is_lowercase(*self) }

    fn is_uppercase(&self) -> bool { is_uppercase(*self) }

    fn is_whitespace(&self) -> bool { is_whitespace(*self) }

    fn is_alphanumeric(&self) -> bool { is_alphanumeric(*self) }

    fn is_control(&self) -> bool { is_control(*self) }

    fn is_digit(&self) -> bool { is_digit(*self) }

    fn is_digit_radix(&self, radix: uint) -> bool { is_digit_radix(*self, radix) }

    fn to_digit(&self, radix: uint) -> Option<uint> { to_digit(*self, radix) }

    fn to_lowercase(&self) -> char { to_lowercase(*self) }

    fn to_uppercase(&self) -> char { to_uppercase(*self) }

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

#[cfg(not(test))]
impl Eq for char {
    #[inline]
    fn eq(&self, other: &char) -> bool { (*self) == (*other) }
}

#[cfg(not(test))]
impl TotalEq for char {}

#[cfg(not(test))]
impl Ord for char {
    #[inline]
    fn lt(&self, other: &char) -> bool { *self < *other }
}

#[cfg(not(test))]
impl TotalOrd for char {
    fn cmp(&self, other: &char) -> Ordering {
        (*self as u32).cmp(&(*other as u32))
    }
}

#[cfg(not(test))]
impl Default for char {
    #[inline]
    fn default() -> char { '\x00' }
}

#[cfg(test)]
mod test {
    use super::{escape_unicode, escape_default};

    use char::Char;
    use slice::ImmutableVector;
    use option::{Some, None};
    use realstd::strbuf::StrBuf;
    use realstd::str::{Str, StrAllocating};

    #[test]
    fn test_is_lowercase() {
        assert!('a'.is_lowercase());
        assert!('ö'.is_lowercase());
        assert!('ß'.is_lowercase());
        assert!(!'Ü'.is_lowercase());
        assert!(!'P'.is_lowercase());
    }

    #[test]
    fn test_is_uppercase() {
        assert!(!'h'.is_uppercase());
        assert!(!'ä'.is_uppercase());
        assert!(!'ß'.is_uppercase());
        assert!('Ö'.is_uppercase());
        assert!('T'.is_uppercase());
    }

    #[test]
    fn test_is_whitespace() {
        assert!(' '.is_whitespace());
        assert!('\u2007'.is_whitespace());
        assert!('\t'.is_whitespace());
        assert!('\n'.is_whitespace());
        assert!(!'a'.is_whitespace());
        assert!(!'_'.is_whitespace());
        assert!(!'\u0000'.is_whitespace());
    }

    #[test]
    fn test_to_digit() {
        assert_eq!('0'.to_digit(10u), Some(0u));
        assert_eq!('1'.to_digit(2u), Some(1u));
        assert_eq!('2'.to_digit(3u), Some(2u));
        assert_eq!('9'.to_digit(10u), Some(9u));
        assert_eq!('a'.to_digit(16u), Some(10u));
        assert_eq!('A'.to_digit(16u), Some(10u));
        assert_eq!('b'.to_digit(16u), Some(11u));
        assert_eq!('B'.to_digit(16u), Some(11u));
        assert_eq!('z'.to_digit(36u), Some(35u));
        assert_eq!('Z'.to_digit(36u), Some(35u));
        assert_eq!(' '.to_digit(10u), None);
        assert_eq!('$'.to_digit(36u), None);
    }

    #[test]
    fn test_to_lowercase() {
        assert_eq!('A'.to_lowercase(), 'a');
        assert_eq!('Ö'.to_lowercase(), 'ö');
        assert_eq!('ß'.to_lowercase(), 'ß');
        assert_eq!('Ü'.to_lowercase(), 'ü');
        assert_eq!('💩'.to_lowercase(), '💩');
        assert_eq!('Σ'.to_lowercase(), 'σ');
        assert_eq!('Τ'.to_lowercase(), 'τ');
        assert_eq!('Ι'.to_lowercase(), 'ι');
        assert_eq!('Γ'.to_lowercase(), 'γ');
        assert_eq!('Μ'.to_lowercase(), 'μ');
        assert_eq!('Α'.to_lowercase(), 'α');
        assert_eq!('Σ'.to_lowercase(), 'σ');
    }

    #[test]
    fn test_to_uppercase() {
        assert_eq!('a'.to_uppercase(), 'A');
        assert_eq!('ö'.to_uppercase(), 'Ö');
        assert_eq!('ß'.to_uppercase(), 'ß'); // not ẞ: Latin capital letter sharp s
        assert_eq!('ü'.to_uppercase(), 'Ü');
        assert_eq!('💩'.to_uppercase(), '💩');

        assert_eq!('σ'.to_uppercase(), 'Σ');
        assert_eq!('τ'.to_uppercase(), 'Τ');
        assert_eq!('ι'.to_uppercase(), 'Ι');
        assert_eq!('γ'.to_uppercase(), 'Γ');
        assert_eq!('μ'.to_uppercase(), 'Μ');
        assert_eq!('α'.to_uppercase(), 'Α');
        assert_eq!('ς'.to_uppercase(), 'Σ');
    }

    #[test]
    fn test_is_control() {
        assert!('\u0000'.is_control());
        assert!('\u0003'.is_control());
        assert!('\u0006'.is_control());
        assert!('\u0009'.is_control());
        assert!('\u007f'.is_control());
        assert!('\u0092'.is_control());
        assert!(!'\u0020'.is_control());
        assert!(!'\u0055'.is_control());
        assert!(!'\u0068'.is_control());
    }

    #[test]
    fn test_is_digit() {
       assert!('2'.is_digit());
       assert!('7'.is_digit());
       assert!(!'c'.is_digit());
       assert!(!'i'.is_digit());
       assert!(!'z'.is_digit());
       assert!(!'Q'.is_digit());
    }

    #[test]
    fn test_escape_default() {
        fn string(c: char) -> StrBuf {
            let mut result = StrBuf::new();
            escape_default(c, |c| { result.push_char(c); });
            return result;
        }
        let s = string('\n');
        assert_eq!(s.as_slice(), "\\n");
        let s = string('\r');
        assert_eq!(s.as_slice(), "\\r");
        let s = string('\'');
        assert_eq!(s.as_slice(), "\\'");
        let s = string('"');
        assert_eq!(s.as_slice(), "\\\"");
        let s = string(' ');
        assert_eq!(s.as_slice(), " ");
        let s = string('a');
        assert_eq!(s.as_slice(), "a");
        let s = string('~');
        assert_eq!(s.as_slice(), "~");
        let s = string('\x00');
        assert_eq!(s.as_slice(), "\\x00");
        let s = string('\x1f');
        assert_eq!(s.as_slice(), "\\x1f");
        let s = string('\x7f');
        assert_eq!(s.as_slice(), "\\x7f");
        let s = string('\xff');
        assert_eq!(s.as_slice(), "\\xff");
        let s = string('\u011b');
        assert_eq!(s.as_slice(), "\\u011b");
        let s = string('\U0001d4b6');
        assert_eq!(s.as_slice(), "\\U0001d4b6");
    }

    #[test]
    fn test_escape_unicode() {
        fn string(c: char) -> StrBuf {
            let mut result = StrBuf::new();
            escape_unicode(c, |c| { result.push_char(c); });
            return result;
        }
        let s = string('\x00');
        assert_eq!(s.as_slice(), "\\x00");
        let s = string('\n');
        assert_eq!(s.as_slice(), "\\x0a");
        let s = string(' ');
        assert_eq!(s.as_slice(), "\\x20");
        let s = string('a');
        assert_eq!(s.as_slice(), "\\x61");
        let s = string('\u011b');
        assert_eq!(s.as_slice(), "\\u011b");
        let s = string('\U0001d4b6');
        assert_eq!(s.as_slice(), "\\U0001d4b6");
    }

    #[test]
    fn test_to_str() {
        use realstd::to_str::ToStr;
        let s = 't'.to_str();
        assert_eq!(s.as_slice(), "t");
    }

    #[test]
    fn test_encode_utf8() {
        fn check(input: char, expect: &[u8]) {
            let mut buf = [0u8, ..4];
            let n = input.encode_utf8(buf /* as mut slice! */);
            assert_eq!(buf.slice_to(n), expect);
        }

        check('x', [0x78]);
        check('\u00e9', [0xc3, 0xa9]);
        check('\ua66e', [0xea, 0x99, 0xae]);
        check('\U0001f4a9', [0xf0, 0x9f, 0x92, 0xa9]);
    }

    #[test]
    fn test_encode_utf16() {
        fn check(input: char, expect: &[u16]) {
            let mut buf = [0u16, ..2];
            let n = input.encode_utf16(buf /* as mut slice! */);
            assert_eq!(buf.slice_to(n), expect);
        }

        check('x', [0x0078]);
        check('\u00e9', [0x00e9]);
        check('\ua66e', [0xa66e]);
        check('\U0001f4a9', [0xd83d, 0xdca9]);
    }
}
