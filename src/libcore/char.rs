/*
Module: char

Utilities for manipulating the char type
*/

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
       to_digit, to_lowercase, to_uppercase, maybe_digit, cmp;

import is_alphabetic = unicode::derived_property::Alphabetic;
import is_XID_start = unicode::derived_property::XID_Start;
import is_XID_continue = unicode::derived_property::XID_Continue;

/*
Function: is_lowercase

Indicates whether a character is in lower case, defined in terms of the
Unicode General Category 'Ll'.
*/
pure fn is_lowercase(c: char) -> bool {
    ret unicode::general_category::Ll(c);
}

/*
Function: is_uppercase

Indicates whether a character is in upper case, defined in terms of the
Unicode General Category 'Lu'.
*/
pure fn is_uppercase(c: char) -> bool {
    ret unicode::general_category::Lu(c);
}

/*
Function: is_whitespace

Indicates whether a character is whitespace, defined in terms of
the Unicode General Categories 'Zs', 'Zl', 'Zp' and the additional
'Cc'-category control codes in the range [0x09, 0x0d].

*/
pure fn is_whitespace(c: char) -> bool {
    ret ('\x09' <= c && c <= '\x0d')
        || unicode::general_category::Zs(c)
        || unicode::general_category::Zl(c)
        || unicode::general_category::Zp(c);
}

/*
Function: is_alphanumeric

Indicates whether a character is alphanumeric, defined in terms of
the Unicode General Categories 'Nd', 'Nl', 'No' and the Derived
Core Property 'Alphabetic'.

*/

pure fn is_alphanumeric(c: char) -> bool {
    ret unicode::derived_property::Alphabetic(c) ||
        unicode::general_category::Nd(c) ||
        unicode::general_category::Nl(c) ||
        unicode::general_category::No(c);
}


/*
 Function: to_digit

 Convert a char to the corresponding digit.

 Parameters:
   c - a char, either '0' to '9', 'a' to 'z' or 'A' to 'Z'

 Returns:
   If `c` is between '0' and '9', the corresponding value between 0 and 9.
 If `c` is 'a' or 'A', 10. If `c` is 'b' or 'B', 11, etc.

 Safety note:
   This function fails if `c` is not a valid char
*/
pure fn to_digit(c: char) -> u8 unsafe {
    alt maybe_digit(c) {
      option::some(x) { x }
      option::none. { fail; }
    }
}

/*
 Function: to_digit

 Convert a char to the corresponding digit. Returns none when the
 character is not a valid hexadecimal digit.
*/
pure fn maybe_digit(c: char) -> option::t<u8> {
    alt c {
      '0' to '9' { option::some(c as u8 - ('0' as u8)) }
      'a' to 'z' { option::some(c as u8 + 10u8 - ('a' as u8)) }
      'A' to 'Z' { option::some(c as u8 + 10u8 - ('A' as u8)) }
      _ { option::none }
    }
}

/*
 Function: to_lowercase

 Convert a char to the corresponding lower case.

 FIXME: works only on ASCII
*/
pure fn to_lowercase(c: char) -> char {
    alt c {
      'A' to 'Z' { ((c as u8) + 32u8) as char }
      _ { c }
    }
}

/*
 Function: to_uppercase

 Convert a char to the corresponding upper case.

 FIXME: works only on ASCII
*/
pure fn to_uppercase(c: char) -> char {
    alt c {
      'a' to 'z' { ((c as u8) - 32u8) as char }
      _ { c }
    }
}

/*
 Function: cmp

 Compare two chars.

 Parameters:
  a - a char
  b - a char

 Returns:
  -1 if a<b, 0 if a==b, +1 if a>b
*/
pure fn cmp(a: char, b: char) -> int {
    ret  if b > a { -1 }
    else if b < a { 1 }
    else { 0 }
}
