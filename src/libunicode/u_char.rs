// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unicode-intensive `char` methods along with the `core` methods.
//!
//! These methods implement functionality for `char` that requires knowledge of
//! Unicode definitions, including normalization, categorization, and display information.

use core::char;
use core::char::CharExt as C;
use core::option::Option;
use tables::{derived_property, property, general_category, conversions, charwidth};

/// Functionality for manipulating `char`.
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
    fn escape_unicode(self) -> char::EscapeUnicode;

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
    fn escape_default(self) -> char::EscapeDefault;

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
    #[unstable = "pending decision about Iterator/Writer/Reader"]
    fn encode_utf8(self, dst: &mut [u8]) -> Option<uint>;

    /// Encodes this character as UTF-16 into the provided `u16` buffer,
    /// and then returns the number of `u16`s written.
    ///
    /// If the buffer is not large enough, nothing will be written into it
    /// and a `None` will be returned.
    #[unstable = "pending decision about Iterator/Writer/Reader"]
    fn encode_utf16(self, dst: &mut [u16]) -> Option<uint>;

    /// Returns whether the specified character is considered a Unicode
    /// alphabetic code point.
    #[stable]
    fn is_alphabetic(self) -> bool;

    /// Returns whether the specified character satisfies the 'XID_Start'
    /// Unicode property.
    ///
    /// 'XID_Start' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to ID_Start but modified for closure under NFKx.
    #[unstable = "mainly needed for compiler internals"]
    fn is_xid_start(self) -> bool;

    /// Returns whether the specified `char` satisfies the 'XID_Continue'
    /// Unicode property.
    ///
    /// 'XID_Continue' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to 'ID_Continue' but modified for closure under NFKx.
    #[unstable = "mainly needed for compiler internals"]
    fn is_xid_continue(self) -> bool;

    /// Indicates whether a character is in lowercase.
    ///
    /// This is defined according to the terms of the Unicode Derived Core
    /// Property `Lowercase`.
    #[stable]
    fn is_lowercase(self) -> bool;

    /// Indicates whether a character is in uppercase.
    ///
    /// This is defined according to the terms of the Unicode Derived Core
    /// Property `Uppercase`.
    #[stable]
    fn is_uppercase(self) -> bool;

    /// Indicates whether a character is whitespace.
    ///
    /// Whitespace is defined in terms of the Unicode Property `White_Space`.
    #[stable]
    fn is_whitespace(self) -> bool;

    /// Indicates whether a character is alphanumeric.
    ///
    /// Alphanumericness is defined in terms of the Unicode General Categories
    /// 'Nd', 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
    #[stable]
    fn is_alphanumeric(self) -> bool;

    /// Indicates whether a character is a control code point.
    ///
    /// Control code points are defined in terms of the Unicode General
    /// Category `Cc`.
    #[stable]
    fn is_control(self) -> bool;

    /// Indicates whether the character is numeric (Nd, Nl, or No).
    #[stable]
    fn is_numeric(self) -> bool;

    /// Converts a character to its lowercase equivalent.
    ///
    /// The case-folding performed is the common or simple mapping. See
    /// `to_uppercase()` for references and more information.
    ///
    /// # Return value
    ///
    /// Returns the lowercase equivalent of the character, or the character
    /// itself if no conversion is possible.
    #[unstable = "pending case transformation decisions"]
    fn to_lowercase(self) -> char;

    /// Converts a character to its uppercase equivalent.
    ///
    /// The case-folding performed is the common or simple mapping: it maps
    /// one Unicode codepoint (one character in Rust) to its uppercase
    /// equivalent according to the Unicode database [1]. The additional
    /// [`SpecialCasing.txt`] is not considered here, as it expands to multiple
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
    /// [`SpecialCasing`.txt`]: ftp://ftp.unicode.org/Public/UNIDATA/SpecialCasing.txt
    ///
    /// [2]: http://www.unicode.org/versions/Unicode4.0.0/ch03.pdf#G33992
    #[unstable = "pending case transformation decisions"]
    fn to_uppercase(self) -> char;

    /// Returns this character's displayed width in columns, or `None` if it is a
    /// control character other than `'\x00'`.
    ///
    /// `is_cjk` determines behavior for characters in the Ambiguous category:
    /// if `is_cjk` is `true`, these are 2 columns wide; otherwise, they are 1.
    /// In CJK contexts, `is_cjk` should be `true`, else it should be `false`.
    /// [Unicode Standard Annex #11](http://www.unicode.org/reports/tr11/)
    /// recommends that these characters be treated as 1 column (i.e.,
    /// `is_cjk` = `false`) if the context cannot be reliably determined.
    #[unstable = "needs expert opinion. is_cjk flag stands out as ugly"]
    fn width(self, is_cjk: bool) -> Option<uint>;
}

#[stable]
impl CharExt for char {
    #[unstable = "pending integer conventions"]
    fn is_digit(self, radix: uint) -> bool { C::is_digit(self, radix) }
    #[unstable = "pending integer conventions"]
    fn to_digit(self, radix: uint) -> Option<uint> { C::to_digit(self, radix) }
    #[stable]
    fn escape_unicode(self) -> char::EscapeUnicode { C::escape_unicode(self) }
    #[stable]
    fn escape_default(self) -> char::EscapeDefault { C::escape_default(self) }
    #[stable]
    fn len_utf8(self) -> uint { C::len_utf8(self) }
    #[stable]
    fn len_utf16(self) -> uint { C::len_utf16(self) }
    #[unstable = "pending decision about Iterator/Writer/Reader"]
    fn encode_utf8(self, dst: &mut [u8]) -> Option<uint> { C::encode_utf8(self, dst) }
    #[unstable = "pending decision about Iterator/Writer/Reader"]
    fn encode_utf16(self, dst: &mut [u16]) -> Option<uint> { C::encode_utf16(self, dst) }

    #[stable]
    fn is_alphabetic(self) -> bool {
        match self {
            'a' ... 'z' | 'A' ... 'Z' => true,
            c if c > '\x7f' => derived_property::Alphabetic(c),
            _ => false
        }
    }

    #[unstable = "mainly needed for compiler internals"]
    fn is_xid_start(self) -> bool { derived_property::XID_Start(self) }

    #[unstable = "mainly needed for compiler internals"]
    fn is_xid_continue(self) -> bool { derived_property::XID_Continue(self) }

    #[stable]
    fn is_lowercase(self) -> bool {
        match self {
            'a' ... 'z' => true,
            c if c > '\x7f' => derived_property::Lowercase(c),
            _ => false
        }
    }

    #[stable]
    fn is_uppercase(self) -> bool {
        match self {
            'A' ... 'Z' => true,
            c if c > '\x7f' => derived_property::Uppercase(c),
            _ => false
        }
    }

    #[stable]
    fn is_whitespace(self) -> bool {
        match self {
            ' ' | '\x09' ... '\x0d' => true,
            c if c > '\x7f' => property::White_Space(c),
            _ => false
        }
    }

    #[stable]
    fn is_alphanumeric(self) -> bool {
        self.is_alphabetic() || self.is_numeric()
    }

    #[stable]
    fn is_control(self) -> bool { general_category::Cc(self) }

    #[stable]
    fn is_numeric(self) -> bool {
        match self {
            '0' ... '9' => true,
            c if c > '\x7f' => general_category::N(c),
            _ => false
        }
    }

    #[unstable = "pending case transformation decisions"]
    fn to_lowercase(self) -> char { conversions::to_lower(self) }

    #[unstable = "pending case transformation decisions"]
    fn to_uppercase(self) -> char { conversions::to_upper(self) }

    #[unstable = "needs expert opinion. is_cjk flag stands out as ugly"]
    fn width(self, is_cjk: bool) -> Option<uint> { charwidth::width(self, is_cjk) }
}
