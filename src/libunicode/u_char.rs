// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Unicode-intensive `char` methods.
 *
 * These methods implement functionality for `char` that requires knowledge of
 * Unicode definitions, including normalization, categorization, and display information.
 */

use core::option::Option;
use tables::{derived_property, property, general_category, conversions, charwidth};

/// Returns whether the specified `char` is considered a Unicode alphabetic
/// code point
pub fn is_alphabetic(c: char) -> bool {
    match c {
        'a' ... 'z' | 'A' ... 'Z' => true,
        c if c > '\x7f' => derived_property::Alphabetic(c),
        _ => false
    }
}

/// Returns whether the specified `char` satisfies the 'XID_Start' Unicode property
///
/// 'XID_Start' is a Unicode Derived Property specified in
/// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
/// mostly similar to ID_Start but modified for closure under NFKx.
#[allow(non_snake_case)]
pub fn is_XID_start(c: char) -> bool    { derived_property::XID_Start(c) }

/// Returns whether the specified `char` satisfies the 'XID_Continue' Unicode property
///
/// 'XID_Continue' is a Unicode Derived Property specified in
/// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
/// mostly similar to 'ID_Continue' but modified for closure under NFKx.
#[allow(non_snake_case)]
pub fn is_XID_continue(c: char) -> bool { derived_property::XID_Continue(c) }

///
/// Indicates whether a `char` is in lower case
///
/// This is defined according to the terms of the Unicode Derived Core Property 'Lowercase'.
///
#[inline]
pub fn is_lowercase(c: char) -> bool {
    match c {
        'a' ... 'z' => true,
        c if c > '\x7f' => derived_property::Lowercase(c),
        _ => false
    }
}

///
/// Indicates whether a `char` is in upper case
///
/// This is defined according to the terms of the Unicode Derived Core Property 'Uppercase'.
///
#[inline]
pub fn is_uppercase(c: char) -> bool {
    match c {
        'A' ... 'Z' => true,
        c if c > '\x7f' => derived_property::Uppercase(c),
        _ => false
    }
}

///
/// Indicates whether a `char` is whitespace
///
/// Whitespace is defined in terms of the Unicode Property 'White_Space'.
///
#[inline]
pub fn is_whitespace(c: char) -> bool {
    match c {
        ' ' | '\x09' ... '\x0d' => true,
        c if c > '\x7f' => property::White_Space(c),
        _ => false
    }
}

///
/// Indicates whether a `char` is alphanumeric
///
/// Alphanumericness is defined in terms of the Unicode General Categories
/// 'Nd', 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
///
#[inline]
pub fn is_alphanumeric(c: char) -> bool {
    is_alphabetic(c)
        || is_digit(c)
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
    match c {
        '0' ... '9' => true,
        c if c > '\x7f' => general_category::N(c),
        _ => false
    }
}

/// Convert a char to its uppercase equivalent
///
/// The case-folding performed is the common or simple mapping:
/// it maps one Unicode codepoint (one char in Rust) to its uppercase equivalent according
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

/// Returns this character's displayed width in columns, or `None` if it is a
/// control character other than `'\x00'`.
///
/// `is_cjk` determines behavior for characters in the Ambiguous category:
/// if `is_cjk` is `true`, these are 2 columns wide; otherwise, they are 1.
/// In CJK contexts, `is_cjk` should be `true`, else it should be `false`.
/// [Unicode Standard Annex #11](http://www.unicode.org/reports/tr11/)
/// recommends that these characters be treated as 1 column (i.e.,
/// `is_cjk` = `false`) if the context cannot be reliably determined.
pub fn width(c: char, is_cjk: bool) -> Option<uint> {
    charwidth::width(c, is_cjk)
}

/// Useful functions for Unicode characters.
pub trait UnicodeChar {
    /// Returns whether the specified character is considered a Unicode
    /// alphabetic code point.
    fn is_alphabetic(&self) -> bool;

    /// Returns whether the specified character satisfies the 'XID_Start'
    /// Unicode property.
    ///
    /// 'XID_Start' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to ID_Start but modified for closure under NFKx.
    #[allow(non_snake_case)]
    fn is_XID_start(&self) -> bool;

    /// Returns whether the specified `char` satisfies the 'XID_Continue'
    /// Unicode property.
    ///
    /// 'XID_Continue' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to 'ID_Continue' but modified for closure under NFKx.
    #[allow(non_snake_case)]
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
    fn to_uppercase(&self) -> char;

    /// Returns this character's displayed width in columns, or `None` if it is a
    /// control character other than `'\x00'`.
    ///
    /// `is_cjk` determines behavior for characters in the Ambiguous category:
    /// if `is_cjk` is `true`, these are 2 columns wide; otherwise, they are 1.
    /// In CJK contexts, `is_cjk` should be `true`, else it should be `false`.
    /// [Unicode Standard Annex #11](http://www.unicode.org/reports/tr11/)
    /// recommends that these characters be treated as 1 column (i.e.,
    /// `is_cjk` = `false`) if the context cannot be reliably determined.
    fn width(&self, is_cjk: bool) -> Option<uint>;
}

impl UnicodeChar for char {
    fn is_alphabetic(&self) -> bool { is_alphabetic(*self) }

    fn is_XID_start(&self) -> bool { is_XID_start(*self) }

    fn is_XID_continue(&self) -> bool { is_XID_continue(*self) }

    fn is_lowercase(&self) -> bool { is_lowercase(*self) }

    fn is_uppercase(&self) -> bool { is_uppercase(*self) }

    fn is_whitespace(&self) -> bool { is_whitespace(*self) }

    fn is_alphanumeric(&self) -> bool { is_alphanumeric(*self) }

    fn is_control(&self) -> bool { is_control(*self) }

    fn is_digit(&self) -> bool { is_digit(*self) }

    fn to_lowercase(&self) -> char { to_lowercase(*self) }

    fn to_uppercase(&self) -> char { to_uppercase(*self) }

    fn width(&self, is_cjk: bool) -> Option<uint> { width(*self, is_cjk) }
}
