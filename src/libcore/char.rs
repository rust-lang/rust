//! Utilities for manipulating the char type

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::Eq;

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

export is_alphabetic,
       is_XID_start, is_XID_continue,
       is_lowercase, is_uppercase,
       is_whitespace, is_alphanumeric,
       is_ascii, is_digit,
       to_digit, cmp,
       escape_default, escape_unicode;

use is_alphabetic = unicode::derived_property::Alphabetic;
use is_XID_start = unicode::derived_property::XID_Start;
use is_XID_continue = unicode::derived_property::XID_Continue;


/**
 * Indicates whether a character is in lower case, defined
 * in terms of the Unicode General Category 'Ll'
 */
pure fn is_lowercase(c: char) -> bool {
    return unicode::general_category::Ll(c);
}

/**
 * Indicates whether a character is in upper case, defined
 * in terms of the Unicode General Category 'Lu'.
 */
pure fn is_uppercase(c: char) -> bool {
    return unicode::general_category::Lu(c);
}

/**
 * Indicates whether a character is whitespace, defined in
 * terms of the Unicode General Categories 'Zs', 'Zl', 'Zp'
 * additional 'Cc'-category control codes in the range [0x09, 0x0d]
 */
pure fn is_whitespace(c: char) -> bool {
    return ('\x09' <= c && c <= '\x0d')
        || unicode::general_category::Zs(c)
        || unicode::general_category::Zl(c)
        || unicode::general_category::Zp(c);
}

/**
 * Indicates whether a character is alphanumeric, defined
 * in terms of the Unicode General Categories 'Nd',
 * 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
 */
pure fn is_alphanumeric(c: char) -> bool {
    return unicode::derived_property::Alphabetic(c) ||
        unicode::general_category::Nd(c) ||
        unicode::general_category::Nl(c) ||
        unicode::general_category::No(c);
}

/// Indicates whether the character is an ASCII character
pure fn is_ascii(c: char) -> bool {
   c - ('\x7F' & c) == '\x00'
}

/// Indicates whether the character is numeric (Nd, Nl, or No)
pure fn is_digit(c: char) -> bool {
    return unicode::general_category::Nd(c) ||
        unicode::general_category::Nl(c) ||
        unicode::general_category::No(c);
}

/**
 * Convert a char to the corresponding digit.
 *
 * # Safety note
 *
 * This function returns none if `c` is not a valid char
 *
 * # Return value
 *
 * If `c` is between '0' and '9', the corresponding value
 * between 0 and 9. If `c` is 'a' or 'A', 10. If `c` is
 * 'b' or 'B', 11, etc. Returns none if the char does not
 * refer to a digit in the given radix.
 */
pure fn to_digit(c: char, radix: uint) -> Option<uint> {
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
 * Return the hexadecimal unicode escape of a char.
 *
 * The rules are as follows:
 *
 *   - chars in [0,0xff] get 2-digit escapes: `\\xNN`
 *   - chars in [0x100,0xffff] get 4-digit escapes: `\\uNNNN`
 *   - chars above 0x10000 get 8-digit escapes: `\\UNNNNNNNN`
 */
fn escape_unicode(c: char) -> ~str {
    let s = u32::to_str(c as u32, 16u);
    let (c, pad) = (if c <= '\xff' { ('x', 2u) }
                    else if c <= '\uffff' { ('u', 4u) }
                    else { ('U', 8u) });
    assert str::len(s) <= pad;
    let mut out = ~"\\";
    str::push_str(out, str::from_char(c));
    for uint::range(str::len(s), pad) |_i| { str::push_str(out, ~"0"); }
    str::push_str(out, s);
    move out
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
fn escape_default(c: char) -> ~str {
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
pure fn cmp(a: char, b: char) -> int {
    return  if b > a { -1 }
    else if b < a { 1 }
    else { 0 }
}

impl char: Eq {
    pure fn eq(&&other: char) -> bool { self == other }
    pure fn ne(&&other: char) -> bool { self != other }
}

#[test]
fn test_is_lowercase() {
    assert is_lowercase('a');
    assert is_lowercase('ö');
    assert is_lowercase('ß');
    assert !is_lowercase('Ü');
    assert !is_lowercase('P');
}

#[test]
fn test_is_uppercase() {
    assert !is_uppercase('h');
    assert !is_uppercase('ä');
    assert !is_uppercase('ß');
    assert is_uppercase('Ö');
    assert is_uppercase('T');
}

#[test]
fn test_is_whitespace() {
    assert is_whitespace(' ');
    assert is_whitespace('\u2007');
    assert is_whitespace('\t');
    assert is_whitespace('\n');

    assert !is_whitespace('a');
    assert !is_whitespace('_');
    assert !is_whitespace('\u0000');
}

#[test]
fn test_to_digit() {
    assert to_digit('0', 10u) == Some(0u);
    assert to_digit('1', 2u) == Some(1u);
    assert to_digit('2', 3u) == Some(2u);
    assert to_digit('9', 10u) == Some(9u);
    assert to_digit('a', 16u) == Some(10u);
    assert to_digit('A', 16u) == Some(10u);
    assert to_digit('b', 16u) == Some(11u);
    assert to_digit('B', 16u) == Some(11u);
    assert to_digit('z', 36u) == Some(35u);
    assert to_digit('Z', 36u) == Some(35u);

    assert to_digit(' ', 10u).is_none();
    assert to_digit('$', 36u).is_none();
}

#[test]
fn test_is_ascii() {
   assert str::all(~"banana", char::is_ascii);
   assert ! str::all(~"ประเทศไทย中华Việt Nam", char::is_ascii);
}

#[test]
fn test_is_digit() {
   assert is_digit('2');
   assert is_digit('7');
   assert ! is_digit('c');
   assert ! is_digit('i');
   assert ! is_digit('z');
   assert ! is_digit('Q');
}

#[test]
fn test_escape_default() {
    assert escape_default('\n') == ~"\\n";
    assert escape_default('\r') == ~"\\r";
    assert escape_default('\'') == ~"\\'";
    assert escape_default('"') == ~"\\\"";
    assert escape_default(' ') == ~" ";
    assert escape_default('a') == ~"a";
    assert escape_default('~') == ~"~";
    assert escape_default('\x00') == ~"\\x00";
    assert escape_default('\x1f') == ~"\\x1f";
    assert escape_default('\x7f') == ~"\\x7f";
    assert escape_default('\xff') == ~"\\xff";
    assert escape_default('\u011b') == ~"\\u011b";
    assert escape_default('\U0001d4b6') == ~"\\U0001d4b6";
}


#[test]
fn test_escape_unicode() {
    assert escape_unicode('\x00') == ~"\\x00";
    assert escape_unicode('\n') == ~"\\x0a";
    assert escape_unicode(' ') == ~"\\x20";
    assert escape_unicode('a') == ~"\\x61";
    assert escape_unicode('\u011b') == ~"\\u011b";
    assert escape_unicode('\U0001d4b6') == ~"\\U0001d4b6";
}
