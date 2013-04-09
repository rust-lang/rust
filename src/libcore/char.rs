// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use u32;
use uint;
use unicode;

#[cfg(notest)] use cmp::Eq;

/*
    Lu  Uppercase_Letter    an uppercase letter
    Ll  Lowercase_Letter    a lowercase letter
    Lt  Titlecase_Letter    a digraphic character, with first part uppercase
    Lm  Modifier_Letter     a modifier letter
    Lo  Other_Letter    other letters, including syllables and ideographs
    Mn  Nonspacing_Mark     a nonspacing combining mark (zero advance width)
    Mc  Spacing_Mark    a spacing combining mark (positive advance width)
    Me  Enclosing_Mark  an enclosing combining mark
    Nd  Decimal_Number  a decimal digit
    Nl  Letter_Number   a letterlike numeric character
    No  Other_Number    a numeric character of other type
    Pc  Connector_Punctuation   a connecting punctuation mark, like a tie
    Pd  Dash_Punctuation    a dash or hyphen punctuation mark
    Ps  Open_Punctuation    an opening punctuation mark (of a pair)
    Pe  Close_Punctuation   a closing punctuation mark (of a pair)
    Pi  Initial_Punctuation     an initial quotation mark
    Pf  Final_Punctuation   a final quotation mark
    Po  Other_Punctuation   a punctuation mark of other type
    Sm  Math_Symbol     a symbol of primarily mathematical use
    Sc  Currency_Symbol     a currency sign
    Sk  Modifier_Symbol     a non-letterlike modifier symbol
    So  Other_Symbol    a symbol of other type
    Zs  Space_Separator     a space character (of various non-zero widths)
    Zl  Line_Separator  U+2028 LINE SEPARATOR only
    Zp  Paragraph_Separator     U+2029 PARAGRAPH SEPARATOR only
    Cc  Control     a C0 or C1 control code
    Cf  Format  a format control character
    Cs  Surrogate   a surrogate code point
    Co  Private_Use     a private-use character
    Cn  Unassigned  a reserved unassigned code point or a noncharacter
*/

pub use is_alphabetic = unicode::derived_property::Alphabetic;
pub use is_XID_start = unicode::derived_property::XID_Start;
pub use is_XID_continue = unicode::derived_property::XID_Continue;


/**
 * Indicates whether a character is in lower case, defined
 * in terms of the Unicode General Category 'Ll'
 */
#[inline(always)]
pub fn is_lowercase(c: char) -> bool {
    return unicode::general_category::Ll(c);
}

/**
 * Indicates whether a character is in upper case, defined
 * in terms of the Unicode General Category 'Lu'.
 */
#[inline(always)]
pub fn is_uppercase(c: char) -> bool {
    return unicode::general_category::Lu(c);
}

/**
 * Indicates whether a character is whitespace. Whitespace is defined in
 * terms of the Unicode General Categories 'Zs', 'Zl', 'Zp'
 * additional 'Cc'-category control codes in the range [0x09, 0x0d]
 */
#[inline(always)]
pub fn is_whitespace(c: char) -> bool {
    return ('\x09' <= c && c <= '\x0d')
        || unicode::general_category::Zs(c)
        || unicode::general_category::Zl(c)
        || unicode::general_category::Zp(c);
}

/**
 * Indicates whether a character is alphanumeric. Alphanumericness is
 * defined in terms of the Unicode General Categories 'Nd', 'Nl', 'No'
 * and the Derived Core Property 'Alphabetic'.
 */
#[inline(always)]
pub fn is_alphanumeric(c: char) -> bool {
    return unicode::derived_property::Alphabetic(c) ||
        unicode::general_category::Nd(c) ||
        unicode::general_category::Nl(c) ||
        unicode::general_category::No(c);
}

/// Indicates whether the character is an ASCII character
#[inline(always)]
pub fn is_ascii(c: char) -> bool {
   c - ('\x7F' & c) == '\x00'
}

/// Indicates whether the character is numeric (Nd, Nl, or No)
#[inline(always)]
pub fn is_digit(c: char) -> bool {
    return unicode::general_category::Nd(c) ||
        unicode::general_category::Nl(c) ||
        unicode::general_category::No(c);
}

/**
 * Checks if a character parses as a numeric digit in the given radix.
 * Compared to `is_digit()`, this function only recognizes the ascii
 * characters `0-9`, `a-z` and `A-Z`.
 *
 * Returns `true` if `c` is a valid digit under `radix`, and `false`
 * otherwise.
 *
 * Fails if given a `radix` > 36.
 *
 * Note: This just wraps `to_digit()`.
 */
#[inline(always)]
pub fn is_digit_radix(c: char, radix: uint) -> bool {
    match to_digit(c, radix) {
        Some(_) => true,
        None    => false
    }
}

/**
 * Convert a char to the corresponding digit.
 *
 * # Return value
 *
 * If `c` is between '0' and '9', the corresponding value
 * between 0 and 9. If `c` is 'a' or 'A', 10. If `c` is
 * 'b' or 'B', 11, etc. Returns none if the char does not
 * refer to a digit in the given radix.
 *
 * # Failure
 * Fails if given a `radix` outside the range `[0..36]`.
 */
#[inline]
pub fn to_digit(c: char, radix: uint) -> Option<uint> {
    if radix > 36 {
        fail!(fmt!("to_digit: radix %? is to high (maximum 36)", radix));
    }
    let val = match c {
      '0' .. '9' => c as uint - ('0' as uint),
      'a' .. 'z' => c as uint + 10u - ('a' as uint),
      'A' .. 'Z' => c as uint + 10u - ('A' as uint),
      _ => return None
    };
    if val < radix { Some(val) }
    else { None }
}

/**
 * Converts a number to the ascii character representing it.
 *
 * Returns `Some(char)` if `num` represents one digit under `radix`,
 * using one character of `0-9` or `a-z`, or `None` if it doesn't.
 *
 * Fails if given an `radix` > 36.
 */
#[inline]
pub fn from_digit(num: uint, radix: uint) -> Option<char> {
    if radix > 36 {
        fail!(fmt!("from_digit: radix %? is to high (maximum 36)", num));
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

/**
 * Return the hexadecimal unicode escape of a char.
 *
 * The rules are as follows:
 *
 *   - chars in [0,0xff] get 2-digit escapes: `\\xNN`
 *   - chars in [0x100,0xffff] get 4-digit escapes: `\\uNNNN`
 *   - chars above 0x10000 get 8-digit escapes: `\\UNNNNNNNN`
 */
pub fn escape_unicode(c: char) -> ~str {
    let s = u32::to_str_radix(c as u32, 16u);
    let (c, pad) = (if c <= '\xff' { ('x', 2u) }
                    else if c <= '\uffff' { ('u', 4u) }
                    else { ('U', 8u) });
    assert!(str::len(s) <= pad);
    let mut out = ~"\\";
    str::push_str(&mut out, str::from_char(c));
    for uint::range(str::len(s), pad) |_i|
        { str::push_str(&mut out, ~"0"); }
    str::push_str(&mut out, s);
    out
}

/**
 * Return a 'default' ASCII and C++11-like char-literal escape of a char.
 *
 * The default is chosen with a bias toward producing literals that are
 * legal in a variety of languages, including C++11 and similar C-family
 * languages. The exact rules are:
 *
 *   - Tab, CR and LF are escaped as '\t', '\r' and '\n' respectively.
 *   - Single-quote, double-quote and backslash chars are backslash-escaped.
 *   - Any other chars in the range [0x20,0x7e] are not escaped.
 *   - Any other chars are given hex unicode escapes; see `escape_unicode`.
 */
pub fn escape_default(c: char) -> ~str {
    match c {
      '\t' => ~"\\t",
      '\r' => ~"\\r",
      '\n' => ~"\\n",
      '\\' => ~"\\\\",
      '\'' => ~"\\'",
      '"'  => ~"\\\"",
      '\x20' .. '\x7e' => str::from_char(c),
      _ => escape_unicode(c)
    }
}

/**
 * Compare two chars
 *
 * # Return value
 *
 * -1 if a < b, 0 if a == b, +1 if a > b
 */
#[inline(always)]
pub fn cmp(a: char, b: char) -> int {
    return  if b > a { -1 }
    else if b < a { 1 }
    else { 0 }
}

#[cfg(notest)]
impl Eq for char {
    fn eq(&self, other: &char) -> bool { (*self) == (*other) }
    fn ne(&self, other: &char) -> bool { (*self) != (*other) }
}

#[test]
fn test_is_lowercase() {
    assert!(is_lowercase('a'));
    assert!(is_lowercase('ö'));
    assert!(is_lowercase('ß'));
    assert!(!is_lowercase('Ü'));
    assert!(!is_lowercase('P'));
}

#[test]
fn test_is_uppercase() {
    assert!(!is_uppercase('h'));
    assert!(!is_uppercase('ä'));
    assert!(!is_uppercase('ß'));
    assert!(is_uppercase('Ö'));
    assert!(is_uppercase('T'));
}

#[test]
fn test_is_whitespace() {
    assert!(is_whitespace(' '));
    assert!(is_whitespace('\u2007'));
    assert!(is_whitespace('\t'));
    assert!(is_whitespace('\n'));

    assert!(!is_whitespace('a'));
    assert!(!is_whitespace('_'));
    assert!(!is_whitespace('\u0000'));
}

#[test]
fn test_to_digit() {
    assert_eq!(to_digit('0', 10u), Some(0u));
    assert_eq!(to_digit('1', 2u), Some(1u));
    assert_eq!(to_digit('2', 3u), Some(2u));
    assert_eq!(to_digit('9', 10u), Some(9u));
    assert_eq!(to_digit('a', 16u), Some(10u));
    assert_eq!(to_digit('A', 16u), Some(10u));
    assert_eq!(to_digit('b', 16u), Some(11u));
    assert_eq!(to_digit('B', 16u), Some(11u));
    assert_eq!(to_digit('z', 36u), Some(35u));
    assert_eq!(to_digit('Z', 36u), Some(35u));

    assert!(to_digit(' ', 10u).is_none());
    assert!(to_digit('$', 36u).is_none());
}

#[test]
fn test_is_ascii() {
   assert!(str::all(~"banana", is_ascii));
   assert!(! str::all(~"ประเทศไทย中华Việt Nam", is_ascii));
}

#[test]
fn test_is_digit() {
   assert!(is_digit('2'));
   assert!(is_digit('7'));
   assert!(! is_digit('c'));
   assert!(! is_digit('i'));
   assert!(! is_digit('z'));
   assert!(! is_digit('Q'));
}

#[test]
fn test_escape_default() {
    assert_eq!(escape_default('\n'), ~"\\n");
    assert_eq!(escape_default('\r'), ~"\\r");
    assert_eq!(escape_default('\''), ~"\\'");
    assert_eq!(escape_default('"'), ~"\\\"");
    assert_eq!(escape_default(' '), ~" ");
    assert_eq!(escape_default('a'), ~"a");
    assert_eq!(escape_default('~'), ~"~");
    assert_eq!(escape_default('\x00'), ~"\\x00");
    assert_eq!(escape_default('\x1f'), ~"\\x1f");
    assert_eq!(escape_default('\x7f'), ~"\\x7f");
    assert_eq!(escape_default('\xff'), ~"\\xff");
    assert_eq!(escape_default('\u011b'), ~"\\u011b");
    assert_eq!(escape_default('\U0001d4b6'), ~"\\U0001d4b6");
}


#[test]
fn test_escape_unicode() {
    assert_eq!(escape_unicode('\x00'), ~"\\x00");
    assert_eq!(escape_unicode('\n'), ~"\\x0a");
    assert_eq!(escape_unicode(' '), ~"\\x20");
    assert_eq!(escape_unicode('a'), ~"\\x61");
    assert_eq!(escape_unicode('\u011b'), ~"\\u011b");
    assert_eq!(escape_unicode('\U0001d4b6'), ~"\\U0001d4b6");
}
