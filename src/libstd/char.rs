// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Utilities for manipulating the char type

use option::{None, Option, Some};
use str;
use str::{StrSlice, OwnedStr};
use u32;
use uint;
use unicode::{derived_property, general_category};

#[cfg(not(test))] use cmp::{Eq, Ord};
#[cfg(not(test))] use num::Zero;

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

/// Returns whether the specified character is considered a unicode alphabetic
/// character
pub fn is_alphabetic(c: char) -> bool   { derived_property::Alphabetic(c) }
#[allow(missing_doc)]
pub fn is_XID_start(c: char) -> bool    { derived_property::XID_Start(c) }
#[allow(missing_doc)]
pub fn is_XID_continue(c: char) -> bool { derived_property::XID_Continue(c) }

///
/// Indicates whether a character is in lower case, defined
/// in terms of the Unicode General Category 'Ll'
///
#[inline(always)]
pub fn is_lowercase(c: char) -> bool { general_category::Ll(c) }

///
/// Indicates whether a character is in upper case, defined
/// in terms of the Unicode General Category 'Lu'.
///
#[inline(always)]
pub fn is_uppercase(c: char) -> bool { general_category::Lu(c) }

///
/// Indicates whether a character is whitespace. Whitespace is defined in
/// terms of the Unicode General Categories 'Zs', 'Zl', 'Zp'
/// additional 'Cc'-category control codes in the range [0x09, 0x0d]
///
#[inline(always)]
pub fn is_whitespace(c: char) -> bool {
    ('\x09' <= c && c <= '\x0d')
        || general_category::Zs(c)
        || general_category::Zl(c)
        || general_category::Zp(c)
}

///
/// Indicates whether a character is alphanumeric. Alphanumericness is
/// defined in terms of the Unicode General Categories 'Nd', 'Nl', 'No'
/// and the Derived Core Property 'Alphabetic'.
///
#[inline(always)]
pub fn is_alphanumeric(c: char) -> bool {
    derived_property::Alphabetic(c)
        || general_category::Nd(c)
        || general_category::Nl(c)
        || general_category::No(c)
}

/// Indicates whether the character is numeric (Nd, Nl, or No)
#[inline(always)]
pub fn is_digit(c: char) -> bool {
    general_category::Nd(c)
        || general_category::Nl(c)
        || general_category::No(c)
}

///
/// Checks if a character parses as a numeric digit in the given radix.
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
#[inline(always)]
pub fn is_digit_radix(c: char, radix: uint) -> bool {
    match to_digit(c, radix) {
        Some(_) => true,
        None    => false,
    }
}

///
/// Convert a char to the corresponding digit.
///
/// # Return value
///
/// If `c` is between '0' and '9', the corresponding value
/// between 0 and 9. If `c` is 'a' or 'A', 10. If `c` is
/// 'b' or 'B', 11, etc. Returns none if the char does not
/// refer to a digit in the given radix.
///
/// # Failure
///
/// Fails if given a `radix` outside the range `[0..36]`.
///
#[inline]
pub fn to_digit(c: char, radix: uint) -> Option<uint> {
    if radix > 36 {
        fail!("to_digit: radix %? is to high (maximum 36)", radix);
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
/// Converts a number to the character representing it.
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
        fail!("from_digit: radix %? is to high (maximum 36)", num);
    }
    if num < radix {
        if num < 10 {
            Some(('0' as uint + num) as char)
        } else {
            Some(('a' as uint + num - 10u) as char)
        }
    } else {
        None
    }
}

///
/// Return the hexadecimal unicode escape of a char.
///
/// The rules are as follows:
///
/// - chars in [0,0xff] get 2-digit escapes: `\\xNN`
/// - chars in [0x100,0xffff] get 4-digit escapes: `\\uNNNN`
/// - chars above 0x10000 get 8-digit escapes: `\\UNNNNNNNN`
///
pub fn escape_unicode(c: char) -> ~str {
    let s = u32::to_str_radix(c as u32, 16u);
    let (c, pad) = cond!(
        (c <= '\xff')   { ('x', 2u) }
        (c <= '\uffff') { ('u', 4u) }
        _               { ('U', 8u) }
    );
    assert!(s.len() <= pad);
    let mut out = ~"\\";
    out.push_str(str::from_char(c));
    for uint::range(s.len(), pad) |_| {
        out.push_str("0");
    }
    out.push_str(s);
    out
}

///
/// Return a 'default' ASCII and C++11-like char-literal escape of a char.
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
pub fn escape_default(c: char) -> ~str {
    match c {
        '\t' => ~"\\t",
        '\r' => ~"\\r",
        '\n' => ~"\\n",
        '\\' => ~"\\\\",
        '\'' => ~"\\'",
        '"'  => ~"\\\"",
        '\x20' .. '\x7e' => str::from_char(c),
        _ => c.escape_unicode(),
    }
}

/// Returns the amount of bytes this character would need if encoded in utf8
pub fn len_utf8_bytes(c: char) -> uint {
    static MAX_ONE_B:   uint = 128u;
    static MAX_TWO_B:   uint = 2048u;
    static MAX_THREE_B: uint = 65536u;
    static MAX_FOUR_B:  uint = 2097152u;

    let code = c as uint;
    cond!(
        (code < MAX_ONE_B)   { 1u }
        (code < MAX_TWO_B)   { 2u }
        (code < MAX_THREE_B) { 3u }
        (code < MAX_FOUR_B)  { 4u }
        _ { fail!("invalid character!") }
    )
}

#[allow(missing_doc)]
pub trait Char {
    fn is_alphabetic(&self) -> bool;
    fn is_XID_start(&self) -> bool;
    fn is_XID_continue(&self) -> bool;
    fn is_lowercase(&self) -> bool;
    fn is_uppercase(&self) -> bool;
    fn is_whitespace(&self) -> bool;
    fn is_alphanumeric(&self) -> bool;
    fn is_digit(&self) -> bool;
    fn is_digit_radix(&self, radix: uint) -> bool;
    fn to_digit(&self, radix: uint) -> Option<uint>;
    fn from_digit(num: uint, radix: uint) -> Option<char>;
    fn escape_unicode(&self) -> ~str;
    fn escape_default(&self) -> ~str;
    fn len_utf8_bytes(&self) -> uint;
}

impl Char for char {
    fn is_alphabetic(&self) -> bool { is_alphabetic(*self) }

    fn is_XID_start(&self) -> bool { is_XID_start(*self) }

    fn is_XID_continue(&self) -> bool { is_XID_continue(*self) }

    fn is_lowercase(&self) -> bool { is_lowercase(*self) }

    fn is_uppercase(&self) -> bool { is_uppercase(*self) }

    fn is_whitespace(&self) -> bool { is_whitespace(*self) }

    fn is_alphanumeric(&self) -> bool { is_alphanumeric(*self) }

    fn is_digit(&self) -> bool { is_digit(*self) }

    fn is_digit_radix(&self, radix: uint) -> bool { is_digit_radix(*self, radix) }

    fn to_digit(&self, radix: uint) -> Option<uint> { to_digit(*self, radix) }

    fn from_digit(num: uint, radix: uint) -> Option<char> { from_digit(num, radix) }

    fn escape_unicode(&self) -> ~str { escape_unicode(*self) }

    fn escape_default(&self) -> ~str { escape_default(*self) }

    fn len_utf8_bytes(&self) -> uint { len_utf8_bytes(*self) }
}

#[cfg(not(test))]
impl Eq for char {
    #[inline(always)]
    fn eq(&self, other: &char) -> bool { (*self) == (*other) }
    #[inline(always)]
    fn ne(&self, other: &char) -> bool { (*self) != (*other) }
}

#[cfg(not(test))]
impl Ord for char {
    #[inline(always)]
    fn lt(&self, other: &char) -> bool { *self < *other }
    #[inline(always)]
    fn le(&self, other: &char) -> bool { *self <= *other }
    #[inline(always)]
    fn gt(&self, other: &char) -> bool { *self > *other }
    #[inline(always)]
    fn ge(&self, other: &char) -> bool { *self >= *other }
}

#[cfg(not(test))]
impl Zero for char {
    fn zero() -> char { 0 as char }
    fn is_zero(&self) -> bool { *self == 0 as char }
}

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
    assert_eq!('\n'.escape_default(), ~"\\n");
    assert_eq!('\r'.escape_default(), ~"\\r");
    assert_eq!('\''.escape_default(), ~"\\'");
    assert_eq!('"'.escape_default(), ~"\\\"");
    assert_eq!(' '.escape_default(), ~" ");
    assert_eq!('a'.escape_default(), ~"a");
    assert_eq!('~'.escape_default(), ~"~");
    assert_eq!('\x00'.escape_default(), ~"\\x00");
    assert_eq!('\x1f'.escape_default(), ~"\\x1f");
    assert_eq!('\x7f'.escape_default(), ~"\\x7f");
    assert_eq!('\xff'.escape_default(), ~"\\xff");
    assert_eq!('\u011b'.escape_default(), ~"\\u011b");
    assert_eq!('\U0001d4b6'.escape_default(), ~"\\U0001d4b6");
}

#[test]
fn test_escape_unicode() {
    assert_eq!('\x00'.escape_unicode(), ~"\\x00");
    assert_eq!('\n'.escape_unicode(), ~"\\x0a");
    assert_eq!(' '.escape_unicode(), ~"\\x20");
    assert_eq!('a'.escape_unicode(), ~"\\x61");
    assert_eq!('\u011b'.escape_unicode(), ~"\\u011b");
    assert_eq!('\U0001d4b6'.escape_unicode(), ~"\\U0001d4b6");
}
