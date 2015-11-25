// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A character type.
//!
//! The `char` type represents a single character. More specifically, since
//! 'character' isn't a well-defined concept in Unicode, `char` is a '[Unicode
//! scalar value]', which is similar to, but not the same as, a '[Unicode code
//! point]'.
//!
//! [Unicode scalar value]: http://www.unicode.org/glossary/#unicode_scalar_value
//! [Unicode code point]: http://www.unicode.org/glossary/#code_point
//!
//! This module exists for technical reasons, the primary documentation for
//! `char` is directly on [the `char` primitive type](../primitive.char.html)
//! itself.
//!
//! This module is the home of the iterator implementations for the iterators
//! implemented on `char`, as well as some useful constants and conversion
//! functions that convert various types to `char`.

#![stable(feature = "rust1", since = "1.0.0")]

use core::char::CharExt as C;
use core::option::Option::{self, Some, None};
use core::iter::Iterator;
use tables::{derived_property, property, general_category, conversions};

// stable reexports
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::char::{MAX, from_u32, from_u32_unchecked, from_digit, EscapeUnicode, EscapeDefault};

// unstable reexports
#[unstable(feature = "unicode", issue = "27783")]
pub use tables::UNICODE_VERSION;

/// Returns an iterator that yields the lowercase equivalent of a `char`.
///
/// This `struct` is created by the [`to_lowercase()`] method on [`char`]. See
/// its documentation for more.
///
/// [`to_lowercase()`]: primitive.char.html#method.escape_to_lowercase
/// [`char`]: primitive.char.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ToLowercase(CaseMappingIter);

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for ToLowercase {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        self.0.next()
    }
}

/// Returns an iterator that yields the uppercase equivalent of a `char`.
///
/// This `struct` is created by the [`to_uppercase()`] method on [`char`]. See
/// its documentation for more.
///
/// [`to_uppercase()`]: primitive.char.html#method.escape_to_uppercase
/// [`char`]: primitive.char.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ToUppercase(CaseMappingIter);

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for ToUppercase {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        self.0.next()
    }
}


enum CaseMappingIter {
    Three(char, char, char),
    Two(char, char),
    One(char),
    Zero,
}

impl CaseMappingIter {
    fn new(chars: [char; 3]) -> CaseMappingIter {
        if chars[2] == '\0' {
            if chars[1] == '\0' {
                CaseMappingIter::One(chars[0])  // Including if chars[0] == '\0'
            } else {
                CaseMappingIter::Two(chars[0], chars[1])
            }
        } else {
            CaseMappingIter::Three(chars[0], chars[1], chars[2])
        }
    }
}

impl Iterator for CaseMappingIter {
    type Item = char;
    fn next(&mut self) -> Option<char> {
        match *self {
            CaseMappingIter::Three(a, b, c) => {
                *self = CaseMappingIter::Two(b, c);
                Some(a)
            }
            CaseMappingIter::Two(b, c) => {
                *self = CaseMappingIter::One(c);
                Some(b)
            }
            CaseMappingIter::One(c) => {
                *self = CaseMappingIter::Zero;
                Some(c)
            }
            CaseMappingIter::Zero => None,
        }
    }
}

#[lang = "char"]
impl char {
    /// Checks if a `char` is a digit in the given radix.
    ///
    /// A 'radix' here is sometimes also called a 'base'. A radix of two
    /// indicates a binary number, a radix of ten, decimal, and a radix of
    /// sixteen, hexicdecimal, to give some common values. Arbitrary
    /// radicum are supported.
    ///
    /// Compared to `is_numeric()`, this function only recognizes the characters
    /// `0-9`, `a-z` and `A-Z`.
    ///
    /// 'Digit' is defined to be only the following characters:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// For a more comprehensive understanding of 'digit', see [`is_numeric()`][is_numeric].
    ///
    /// [is_numeric]: #method.is_numeric
    ///
    /// # Panics
    ///
    /// Panics if given a radix larger than 36.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let d = '1';
    ///
    /// assert!(d.is_digit(10));
    ///
    /// let d = 'f';
    ///
    /// assert!(d.is_digit(16));
    /// assert!(!d.is_digit(10));
    /// ```
    ///
    /// Passing a large radix, causing a panic:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let result = thread::spawn(|| {
    ///     let d = '1';
    ///
    ///     // this panics
    ///     d.is_digit(37);
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_digit(self, radix: u32) -> bool {
        C::is_digit(self, radix)
    }

    /// Converts a `char` to a digit in the given radix.
    ///
    /// A 'radix' here is sometimes also called a 'base'. A radix of two
    /// indicates a binary number, a radix of ten, decimal, and a radix of
    /// sixteen, hexicdecimal, to give some common values. Arbitrary
    /// radicum are supported.
    ///
    /// 'Digit' is defined to be only the following characters:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// # Failure
    ///
    /// Returns `None` if the `char` does not refer to a digit in the given radix.
    ///
    /// # Panics
    ///
    /// Panics if given a radix larger than 36.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let d = '1';
    ///
    /// assert_eq!(d.to_digit(10), Some(1));
    ///
    /// let d = 'f';
    ///
    /// assert_eq!(d.to_digit(16), Some(15));
    /// ```
    ///
    /// Passing a non-digit results in failure:
    ///
    /// ```
    /// let d = 'f';
    ///
    /// assert_eq!(d.to_digit(10), None);
    ///
    /// let d = 'z';
    ///
    /// assert_eq!(d.to_digit(16), None);
    /// ```
    ///
    /// Passing a large radix, causing a panic:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let result = thread::spawn(|| {
    ///   let d = '1';
    ///
    ///   d.to_digit(37);
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_digit(self, radix: u32) -> Option<u32> {
        C::to_digit(self, radix)
    }

    /// Returns an iterator that yields the hexadecimal Unicode escape of a
    /// character, as `char`s.
    ///
    /// All characters are escaped with Rust syntax of the form `\\u{NNNN}`
    /// where `NNNN` is the shortest hexadecimal representation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// for c in '❤'.escape_unicode() {
    ///     print!("{}", c);
    /// }
    /// println!("");
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// \u{2764}
    /// ```
    ///
    /// Collecting into a `String`:
    ///
    /// ```
    /// let heart: String = '❤'.escape_unicode().collect();
    ///
    /// assert_eq!(heart, r"\u{2764}");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn escape_unicode(self) -> EscapeUnicode {
        C::escape_unicode(self)
    }

    /// Returns an iterator that yields the literal escape code of a `char`.
    ///
    /// The default is chosen with a bias toward producing literals that are
    /// legal in a variety of languages, including C++11 and similar C-family
    /// languages. The exact rules are:
    ///
    /// * Tab is escaped as `\t`.
    /// * Carriage return is escaped as `\r`.
    /// * Line feed is escaped as `\n`.
    /// * Single quote is escaped as `\'`.
    /// * Double quote is escaped as `\"`.
    /// * Backslash is escaped as `\\`.
    /// * Any character in the 'printable ASCII' range `0x20` .. `0x7e`
    ///   inclusive is not escaped.
    /// * All other characters are given hexadecimal Unicode escapes; see
    ///   [`escape_unicode`][escape_unicode].
    ///
    /// [escape_unicode]: #method.escape_unicode
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// for i in '"'.escape_default() {
    ///     println!("{}", i);
    /// }
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// \
    /// "
    /// ```
    ///
    /// Collecting into a `String`:
    ///
    /// ```
    /// let quote: String = '"'.escape_default().collect();
    ///
    /// assert_eq!(quote, "\\\"");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn escape_default(self) -> EscapeDefault {
        C::escape_default(self)
    }

    /// Returns the number of bytes this `char` would need if encoded in UTF-8.
    ///
    /// That number of bytes is always between 1 and 4, inclusive.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let len = 'A'.len_utf8();
    /// assert_eq!(len, 1);
    ///
    /// let len = 'ß'.len_utf8();
    /// assert_eq!(len, 2);
    ///
    /// let len = 'ℝ'.len_utf8();
    /// assert_eq!(len, 3);
    ///
    /// let len = '💣'.len_utf8();
    /// assert_eq!(len, 4);
    /// ```
    ///
    /// The `&str` type guarantees that its contents are UTF-8, and so we can compare the length it
    /// would take if each code point was represented as a `char` vs in the `&str` itself:
    ///
    /// ```
    /// // as chars
    /// let eastern = '東';
    /// let capitol = '京';
    ///
    /// // both can be represented as three bytes
    /// assert_eq!(3, eastern.len_utf8());
    /// assert_eq!(3, capitol.len_utf8());
    ///
    /// // as a &str, these two are encoded in UTF-8
    /// let tokyo = "東京";
    ///
    /// let len = eastern.len_utf8() + capitol.len_utf8();
    ///
    /// // we can see that they take six bytes total...
    /// assert_eq!(6, tokyo.len());
    ///
    /// // ... just like the &str
    /// assert_eq!(len, tokyo.len());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len_utf8(self) -> usize {
        C::len_utf8(self)
    }

    /// Returns the number of 16-bit code units this `char` would need if
    /// encoded in UTF-16.
    ///
    /// See the documentation for [`len_utf8()`][len_utf8] for more explanation
    /// of this concept. This function is a mirror, but for UTF-16 instead of
    /// UTF-8.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let n = 'ß'.len_utf16();
    /// assert_eq!(n, 1);
    ///
    /// let len = '💣'.len_utf16();
    /// assert_eq!(len, 2);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len_utf16(self) -> usize {
        C::len_utf16(self)
    }

    /// Encodes this character as UTF-8 into the provided byte buffer, and then
    /// returns the number of bytes written.
    ///
    /// If the buffer is not large enough, nothing will be written into it and a
    /// `None` will be returned. A buffer of length four is large enough to
    /// encode any `char`.
    ///
    /// # Examples
    ///
    /// In both of these examples, 'ß' takes two bytes to encode.
    ///
    /// ```
    /// #![feature(unicode)]
    ///
    /// let mut b = [0; 2];
    ///
    /// let result = 'ß'.encode_utf8(&mut b);
    ///
    /// assert_eq!(result, Some(2));
    /// ```
    ///
    /// A buffer that's too small:
    ///
    /// ```
    /// #![feature(unicode)]
    ///
    /// let mut b = [0; 1];
    ///
    /// let result = 'ß'.encode_utf8(&mut b);
    ///
    /// assert_eq!(result, None);
    /// ```
    #[unstable(feature = "unicode",
               reason = "pending decision about Iterator/Writer/Reader",
               issue = "27784")]
    #[inline]
    pub fn encode_utf8(self, dst: &mut [u8]) -> Option<usize> {
        C::encode_utf8(self, dst)
    }

    /// Encodes this character as UTF-16 into the provided `u16` buffer, and
    /// then returns the number of `u16`s written.
    ///
    /// If the buffer is not large enough, nothing will be written into it and a
    /// `None` will be returned. A buffer of length 2 is large enough to encode
    /// any `char`.
    ///
    /// # Examples
    ///
    /// In both of these examples, 'ß' takes one `u16` to encode.
    ///
    /// ```
    /// #![feature(unicode)]
    ///
    /// let mut b = [0; 1];
    ///
    /// let result = 'ß'.encode_utf16(&mut b);
    ///
    /// assert_eq!(result, Some(1));
    /// ```
    ///
    /// A buffer that's too small:
    ///
    /// ```
    /// #![feature(unicode)]
    ///
    /// let mut b = [0; 0];
    ///
    /// let result = 'ß'.encode_utf8(&mut b);
    ///
    /// assert_eq!(result, None);
    /// ```
    #[unstable(feature = "unicode",
               reason = "pending decision about Iterator/Writer/Reader",
               issue = "27784")]
    #[inline]
    pub fn encode_utf16(self, dst: &mut [u16]) -> Option<usize> {
        C::encode_utf16(self, dst)
    }

    /// Returns true if this `char` is an alphabetic code point, and false if not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = 'a';
    ///
    /// assert!(c.is_alphabetic());
    ///
    /// let c = '京';
    /// assert!(c.is_alphabetic());
    ///
    /// let c = '💝';
    /// // love is many things, but it is not alphabetic
    /// assert!(!c.is_alphabetic());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_alphabetic(self) -> bool {
        match self {
            'a'...'z' | 'A'...'Z' => true,
            c if c > '\x7f' => derived_property::Alphabetic(c),
            _ => false,
        }
    }

    /// Returns true if this `char` satisfies the 'XID_Start' Unicode property, and false
    /// otherwise.
    ///
    /// 'XID_Start' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to `ID_Start` but modified for closure under `NFKx`.
    #[unstable(feature = "unicode",
               reason = "mainly needed for compiler internals",
               issue = "0")]
    #[inline]
    pub fn is_xid_start(self) -> bool {
        derived_property::XID_Start(self)
    }

    /// Returns true if this `char` satisfies the 'XID_Continue' Unicode property, and false
    /// otherwise.
    ///
    /// 'XID_Continue' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to 'ID_Continue' but modified for closure under NFKx.
    #[unstable(feature = "unicode",
               reason = "mainly needed for compiler internals",
               issue = "0")]
    #[inline]
    pub fn is_xid_continue(self) -> bool {
        derived_property::XID_Continue(self)
    }

    /// Returns true if this `char` is lowercase, and false otherwise.
    ///
    /// 'Lowercase' is defined according to the terms of the Unicode Derived Core
    /// Property `Lowercase`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = 'a';
    /// assert!(c.is_lowercase());
    ///
    /// let c = 'δ';
    /// assert!(c.is_lowercase());
    ///
    /// let c = 'A';
    /// assert!(!c.is_lowercase());
    ///
    /// let c = 'Δ';
    /// assert!(!c.is_lowercase());
    ///
    /// // The various Chinese scripts do not have case, and so:
    /// let c = '中';
    /// assert!(!c.is_lowercase());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_lowercase(self) -> bool {
        match self {
            'a'...'z' => true,
            c if c > '\x7f' => derived_property::Lowercase(c),
            _ => false,
        }
    }

    /// Returns true if this `char` is uppercase, and false otherwise.
    ///
    /// 'Uppercase' is defined according to the terms of the Unicode Derived Core
    /// Property `Uppercase`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = 'a';
    /// assert!(!c.is_uppercase());
    ///
    /// let c = 'δ';
    /// assert!(!c.is_uppercase());
    ///
    /// let c = 'A';
    /// assert!(c.is_uppercase());
    ///
    /// let c = 'Δ';
    /// assert!(c.is_uppercase());
    ///
    /// // The various Chinese scripts do not have case, and so:
    /// let c = '中';
    /// assert!(!c.is_uppercase());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_uppercase(self) -> bool {
        match self {
            'A'...'Z' => true,
            c if c > '\x7f' => derived_property::Uppercase(c),
            _ => false,
        }
    }

    /// Returns true if this `char` is whitespace, and false otherwise.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived Core
    /// Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = ' ';
    /// assert!(c.is_whitespace());
    ///
    /// // a non-breaking space
    /// let c = '\u{A0}';
    /// assert!(c.is_whitespace());
    ///
    /// let c = '越';
    /// assert!(!c.is_whitespace());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_whitespace(self) -> bool {
        match self {
            ' ' | '\x09'...'\x0d' => true,
            c if c > '\x7f' => property::White_Space(c),
            _ => false,
        }
    }

    /// Returns true if this `char` is alphanumeric, and false otherwise.
    ///
    /// 'Alphanumeric'-ness is defined in terms of the Unicode General Categories
    /// 'Nd', 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = '٣';
    /// assert!(c.is_alphanumeric());
    ///
    /// let c = '7';
    /// assert!(c.is_alphanumeric());
    ///
    /// let c = '৬';
    /// assert!(c.is_alphanumeric());
    ///
    /// let c = 'K';
    /// assert!(c.is_alphanumeric());
    ///
    /// let c = 'و';
    /// assert!(c.is_alphanumeric());
    ///
    /// let c = '藏';
    /// assert!(c.is_alphanumeric());
    ///
    /// let c = '¾';
    /// assert!(!c.is_alphanumeric());
    ///
    /// let c = '①';
    /// assert!(!c.is_alphanumeric());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_alphanumeric(self) -> bool {
        self.is_alphabetic() || self.is_numeric()
    }

    /// Returns true if this `char` is a control code point, and false otherwise.
    ///
    /// 'Control code point' is defined in terms of the Unicode General
    /// Category `Cc`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // U+009C, STRING TERMINATOR
    /// let c = '';
    /// assert!(c.is_control());
    ///
    /// let c = 'q';
    /// assert!(!c.is_control());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_control(self) -> bool {
        general_category::Cc(self)
    }

    /// Returns true if this `char` is numeric, and false otherwise.
    ///
    /// 'Numeric'-ness is defined in terms of the Unicode General Categories
    /// 'Nd', 'Nl', 'No'.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = '٣';
    /// assert!(c.is_numeric());
    ///
    /// let c = '7';
    /// assert!(c.is_numeric());
    ///
    /// let c = '৬';
    /// assert!(c.is_numeric());
    ///
    /// let c = 'K';
    /// assert!(!c.is_numeric());
    ///
    /// let c = 'و';
    /// assert!(!c.is_numeric());
    ///
    /// let c = '藏';
    /// assert!(!c.is_numeric());
    ///
    /// let c = '¾';
    /// assert!(!c.is_numeric());
    ///
    /// let c = '①';
    /// assert!(!c.is_numeric());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_numeric(self) -> bool {
        match self {
            '0'...'9' => true,
            c if c > '\x7f' => general_category::N(c),
            _ => false,
        }
    }

    /// Returns an iterator that yields the lowercase equivalent of a `char`.
    ///
    /// If no conversion is possible then an iterator with just the input character is returned.
    ///
    /// This performs complex unconditional mappings with no tailoring: it maps
    /// one Unicode character to its lowercase equivalent according to the
    /// [Unicode database] and the additional complex mappings
    /// [`SpecialCasing.txt`]. Conditional mappings (based on context or
    /// language) are not considered here.
    ///
    /// For a full reference, see [here][reference].
    ///
    /// [Unicode database]: ftp://ftp.unicode.org/Public/UNIDATA/UnicodeData.txt
    ///
    /// [`SpecialCasing.txt`]: ftp://ftp.unicode.org/Public/UNIDATA/SpecialCasing.txt
    ///
    /// [reference]: http://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = 'c';
    ///
    /// assert_eq!(c.to_uppercase().next(), Some('C'));
    ///
    /// // Japanese scripts do not have case, and so:
    /// let c = '山';
    /// assert_eq!(c.to_uppercase().next(), Some('山'));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_lowercase(self) -> ToLowercase {
        ToLowercase(CaseMappingIter::new(conversions::to_lower(self)))
    }

    /// Returns an iterator that yields the uppercase equivalent of a `char`.
    ///
    /// If no conversion is possible then an iterator with just the input character is returned.
    ///
    /// This performs complex unconditional mappings with no tailoring: it maps
    /// one Unicode character to its uppercase equivalent according to the
    /// [Unicode database] and the additional complex mappings
    /// [`SpecialCasing.txt`]. Conditional mappings (based on context or
    /// language) are not considered here.
    ///
    /// For a full reference, see [here][reference].
    ///
    /// [Unicode database]: ftp://ftp.unicode.org/Public/UNIDATA/UnicodeData.txt
    ///
    /// [`SpecialCasing.txt`]: ftp://ftp.unicode.org/Public/UNIDATA/SpecialCasing.txt
    ///
    /// [reference]: http://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = 'c';
    /// assert_eq!(c.to_uppercase().next(), Some('C'));
    ///
    /// // Japanese does not have case, and so:
    /// let c = '山';
    /// assert_eq!(c.to_uppercase().next(), Some('山'));
    /// ```
    ///
    /// In Turkish, the equivalent of 'i' in Latin has five forms instead of two:
    ///
    /// * 'Dotless': I / ı, sometimes written ï
    /// * 'Dotted': İ / i
    ///
    /// Note that the lowercase dotted 'i' is the same as the Latin. Therefore:
    ///
    /// ```
    /// let i = 'i';
    ///
    /// let upper_i = i.to_uppercase().next();
    /// ```
    ///
    /// The value of `upper_i` here relies on the language of the text: if we're
    /// in `en-US`, it should be `Some('I')`, but if we're in `tr_TR`, it should
    /// be `Some('İ')`. `to_uppercase()` does not take this into account, and so:
    ///
    /// ```
    /// let i = 'i';
    ///
    /// let upper_i = i.to_uppercase().next();
    ///
    /// assert_eq!(Some('I'), upper_i);
    /// ```
    ///
    /// holds across languages.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_uppercase(self) -> ToUppercase {
        ToUppercase(CaseMappingIter::new(conversions::to_upper(self)))
    }
}

/// An iterator that decodes UTF-16 encoded code points from an iterator of `u16`s.
#[unstable(feature = "decode_utf16", reason = "recently exposed", issue = "27830")]
#[derive(Clone)]
pub struct DecodeUtf16<I>
    where I: Iterator<Item = u16>
{
    iter: I,
    buf: Option<u16>,
}

/// Create an iterator over the UTF-16 encoded code points in `iterable`,
/// returning unpaired surrogates as `Err`s.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// #![feature(decode_utf16)]
///
/// use std::char::decode_utf16;
///
/// fn main() {
///     // 𝄞mus<invalid>ic<invalid>
///     let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///              0x0073, 0xDD1E, 0x0069, 0x0063,
///              0xD834];
///
///     assert_eq!(decode_utf16(v.iter().cloned()).collect::<Vec<_>>(),
///                vec![Ok('𝄞'),
///                     Ok('m'), Ok('u'), Ok('s'),
///                     Err(0xDD1E),
///                     Ok('i'), Ok('c'),
///                     Err(0xD834)]);
/// }
/// ```
///
/// A lossy decoder can be obtained by replacing `Err` results with the replacement character:
///
/// ```
/// #![feature(decode_utf16)]
///
/// use std::char::{decode_utf16, REPLACEMENT_CHARACTER};
///
/// fn main() {
///     // 𝄞mus<invalid>ic<invalid>
///     let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///              0x0073, 0xDD1E, 0x0069, 0x0063,
///              0xD834];
///
///     assert_eq!(decode_utf16(v.iter().cloned())
///                    .map(|r| r.unwrap_or(REPLACEMENT_CHARACTER))
///                    .collect::<String>(),
///                "𝄞mus�ic�");
/// }
/// ```
#[unstable(feature = "decode_utf16", reason = "recently exposed", issue = "27830")]
#[inline]
pub fn decode_utf16<I: IntoIterator<Item = u16>>(iterable: I) -> DecodeUtf16<I::IntoIter> {
    DecodeUtf16 {
        iter: iterable.into_iter(),
        buf: None,
    }
}

#[unstable(feature = "decode_utf16", reason = "recently exposed", issue = "27830")]
impl<I: Iterator<Item=u16>> Iterator for DecodeUtf16<I> {
    type Item = Result<char, u16>;

    fn next(&mut self) -> Option<Result<char, u16>> {
        let u = match self.buf.take() {
            Some(buf) => buf,
            None => match self.iter.next() {
                Some(u) => u,
                None => return None,
            },
        };

        if u < 0xD800 || 0xDFFF < u {
            // not a surrogate
            Some(Ok(unsafe { from_u32_unchecked(u as u32) }))
        } else if u >= 0xDC00 {
            // a trailing surrogate
            Some(Err(u))
        } else {
            let u2 = match self.iter.next() {
                Some(u2) => u2,
                // eof
                None => return Some(Err(u)),
            };
            if u2 < 0xDC00 || u2 > 0xDFFF {
                // not a trailing surrogate so we're not a valid
                // surrogate pair, so rewind to redecode u2 next time.
                self.buf = Some(u2);
                return Some(Err(u));
            }

            // all ok, so lets decode it.
            let c = (((u - 0xD800) as u32) << 10 | (u2 - 0xDC00) as u32) + 0x1_0000;
            Some(Ok(unsafe { from_u32_unchecked(c) }))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        // we could be entirely valid surrogates (2 elements per
        // char), or entirely non-surrogates (1 element per char)
        (low / 2, high)
    }
}

/// `U+FFFD REPLACEMENT CHARACTER` (�) is used in Unicode to represent a decoding error.
/// It can occur, for example, when giving ill-formed UTF-8 bytes to
/// [`String::from_utf8_lossy`](../string/struct.String.html#method.from_utf8_lossy).
#[unstable(feature = "decode_utf16", reason = "recently added", issue = "27830")]
pub const REPLACEMENT_CHARACTER: char = '\u{FFFD}';
