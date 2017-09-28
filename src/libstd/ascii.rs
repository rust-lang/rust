// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on ASCII strings and characters.
//!
//! Most string operations in Rust act on UTF-8 strings. However, at times it
//! makes more sense to only consider the ASCII character set for a specific
//! operation.
//!
//! The [`AsciiExt`] trait provides methods that allow for character
//! operations that only act on the ASCII subset and leave non-ASCII characters
//! alone.
//!
//! The [`escape_default`] function provides an iterator over the bytes of an
//! escaped version of the character given.
//!
//! [`AsciiExt`]: trait.AsciiExt.html
//! [`escape_default`]: fn.escape_default.html

#![stable(feature = "rust1", since = "1.0.0")]

use fmt;
use ops::Range;
use iter::FusedIterator;

/// Extension methods for ASCII-subset only operations.
///
/// Be aware that operations on seemingly non-ASCII characters can sometimes
/// have unexpected results. Consider this example:
///
/// ```
/// use std::ascii::AsciiExt;
///
/// assert_eq!("café".to_ascii_uppercase(), "CAFÉ");
/// assert_eq!("café".to_ascii_uppercase(), "CAFé");
/// ```
///
/// In the first example, the lowercased string is represented `"cafe\u{301}"`
/// (the last character is an acute accent [combining character]). Unlike the
/// other characters in the string, the combining character will not get mapped
/// to an uppercase variant, resulting in `"CAFE\u{301}"`. In the second
/// example, the lowercased string is represented `"caf\u{e9}"` (the last
/// character is a single Unicode character representing an 'e' with an acute
/// accent). Since the last character is defined outside the scope of ASCII,
/// it will not get mapped to an uppercase variant, resulting in `"CAF\u{e9}"`.
///
/// [combining character]: https://en.wikipedia.org/wiki/Combining_character
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsciiExt {
    /// Container type for copied ASCII characters.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Owned;

    /// Checks if the value is within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let ascii = 'a';
    /// let non_ascii = '❤';
    /// let int_ascii = 97;
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// assert!(int_ascii.is_ascii());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_ascii(&self) -> bool;

    /// Makes a copy of the value in its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// To uppercase ASCII characters in addition to non-ASCII characters, use
    /// [`str::to_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let ascii = 'a';
    /// let non_ascii = '❤';
    /// let int_ascii = 97;
    ///
    /// assert_eq!('A', ascii.to_ascii_uppercase());
    /// assert_eq!('❤', non_ascii.to_ascii_uppercase());
    /// assert_eq!(65, int_ascii.to_ascii_uppercase());
    /// ```
    ///
    /// [`make_ascii_uppercase`]: #tymethod.make_ascii_uppercase
    /// [`str::to_uppercase`]: ../primitive.str.html#method.to_uppercase
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_ascii_uppercase(&self) -> Self::Owned;

    /// Makes a copy of the value in its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// To lowercase ASCII characters in addition to non-ASCII characters, use
    /// [`str::to_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let ascii = 'A';
    /// let non_ascii = '❤';
    /// let int_ascii = 65;
    ///
    /// assert_eq!('a', ascii.to_ascii_lowercase());
    /// assert_eq!('❤', non_ascii.to_ascii_lowercase());
    /// assert_eq!(97, int_ascii.to_ascii_lowercase());
    /// ```
    ///
    /// [`make_ascii_lowercase`]: #tymethod.make_ascii_lowercase
    /// [`str::to_lowercase`]: ../primitive.str.html#method.to_lowercase
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_ascii_lowercase(&self) -> Self::Owned;

    /// Checks that two values are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let ascii1 = 'A';
    /// let ascii2 = 'a';
    /// let ascii3 = 'A';
    /// let ascii4 = 'z';
    ///
    /// assert!(ascii1.eq_ignore_ascii_case(&ascii2));
    /// assert!(ascii1.eq_ignore_ascii_case(&ascii3));
    /// assert!(!ascii1.eq_ignore_ascii_case(&ascii4));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn eq_ignore_ascii_case(&self, other: &Self) -> bool;

    /// Converts this type to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let mut ascii = 'a';
    ///
    /// ascii.make_ascii_uppercase();
    ///
    /// assert_eq!('A', ascii);
    /// ```
    ///
    /// [`to_ascii_uppercase`]: #tymethod.to_ascii_uppercase
    #[stable(feature = "ascii", since = "1.9.0")]
    fn make_ascii_uppercase(&mut self);

    /// Converts this type to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ascii::AsciiExt;
    ///
    /// let mut ascii = 'A';
    ///
    /// ascii.make_ascii_lowercase();
    ///
    /// assert_eq!('a', ascii);
    /// ```
    ///
    /// [`to_ascii_lowercase`]: #tymethod.to_ascii_lowercase
    #[stable(feature = "ascii", since = "1.9.0")]
    fn make_ascii_lowercase(&mut self);

    /// Checks if the value is an ASCII alphabetic character:
    /// U+0041 'A' ... U+005A 'Z' or U+0061 'a' ... U+007A 'z'.
    /// For strings, true if all characters in the string are
    /// ASCII alphabetic.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(A.is_ascii_alphabetic());
    /// assert!(G.is_ascii_alphabetic());
    /// assert!(a.is_ascii_alphabetic());
    /// assert!(g.is_ascii_alphabetic());
    /// assert!(!zero.is_ascii_alphabetic());
    /// assert!(!percent.is_ascii_alphabetic());
    /// assert!(!space.is_ascii_alphabetic());
    /// assert!(!lf.is_ascii_alphabetic());
    /// assert!(!esc.is_ascii_alphabetic());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_alphabetic(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII uppercase character:
    /// U+0041 'A' ... U+005A 'Z'.
    /// For strings, true if all characters in the string are
    /// ASCII uppercase.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(A.is_ascii_uppercase());
    /// assert!(G.is_ascii_uppercase());
    /// assert!(!a.is_ascii_uppercase());
    /// assert!(!g.is_ascii_uppercase());
    /// assert!(!zero.is_ascii_uppercase());
    /// assert!(!percent.is_ascii_uppercase());
    /// assert!(!space.is_ascii_uppercase());
    /// assert!(!lf.is_ascii_uppercase());
    /// assert!(!esc.is_ascii_uppercase());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_uppercase(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII lowercase character:
    /// U+0061 'a' ... U+007A 'z'.
    /// For strings, true if all characters in the string are
    /// ASCII lowercase.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(!A.is_ascii_lowercase());
    /// assert!(!G.is_ascii_lowercase());
    /// assert!(a.is_ascii_lowercase());
    /// assert!(g.is_ascii_lowercase());
    /// assert!(!zero.is_ascii_lowercase());
    /// assert!(!percent.is_ascii_lowercase());
    /// assert!(!space.is_ascii_lowercase());
    /// assert!(!lf.is_ascii_lowercase());
    /// assert!(!esc.is_ascii_lowercase());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_lowercase(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII alphanumeric character:
    /// U+0041 'A' ... U+005A 'Z', U+0061 'a' ... U+007A 'z', or
    /// U+0030 '0' ... U+0039 '9'.
    /// For strings, true if all characters in the string are
    /// ASCII alphanumeric.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(A.is_ascii_alphanumeric());
    /// assert!(G.is_ascii_alphanumeric());
    /// assert!(a.is_ascii_alphanumeric());
    /// assert!(g.is_ascii_alphanumeric());
    /// assert!(zero.is_ascii_alphanumeric());
    /// assert!(!percent.is_ascii_alphanumeric());
    /// assert!(!space.is_ascii_alphanumeric());
    /// assert!(!lf.is_ascii_alphanumeric());
    /// assert!(!esc.is_ascii_alphanumeric());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_alphanumeric(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII decimal digit:
    /// U+0030 '0' ... U+0039 '9'.
    /// For strings, true if all characters in the string are
    /// ASCII digits.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(!A.is_ascii_digit());
    /// assert!(!G.is_ascii_digit());
    /// assert!(!a.is_ascii_digit());
    /// assert!(!g.is_ascii_digit());
    /// assert!(zero.is_ascii_digit());
    /// assert!(!percent.is_ascii_digit());
    /// assert!(!space.is_ascii_digit());
    /// assert!(!lf.is_ascii_digit());
    /// assert!(!esc.is_ascii_digit());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_digit(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII hexadecimal digit:
    /// U+0030 '0' ... U+0039 '9', U+0041 'A' ... U+0046 'F', or
    /// U+0061 'a' ... U+0066 'f'.
    /// For strings, true if all characters in the string are
    /// ASCII hex digits.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(A.is_ascii_hexdigit());
    /// assert!(!G.is_ascii_hexdigit());
    /// assert!(a.is_ascii_hexdigit());
    /// assert!(!g.is_ascii_hexdigit());
    /// assert!(zero.is_ascii_hexdigit());
    /// assert!(!percent.is_ascii_hexdigit());
    /// assert!(!space.is_ascii_hexdigit());
    /// assert!(!lf.is_ascii_hexdigit());
    /// assert!(!esc.is_ascii_hexdigit());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_hexdigit(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII punctuation character:
    ///
    /// U+0021 ... U+002F `! " # $ % & ' ( ) * + , - . /`
    /// U+003A ... U+0040 `: ; < = > ? @`
    /// U+005B ... U+0060 ``[ \\ ] ^ _ ` ``
    /// U+007B ... U+007E `{ | } ~`
    ///
    /// For strings, true if all characters in the string are
    /// ASCII punctuation.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(!A.is_ascii_punctuation());
    /// assert!(!G.is_ascii_punctuation());
    /// assert!(!a.is_ascii_punctuation());
    /// assert!(!g.is_ascii_punctuation());
    /// assert!(!zero.is_ascii_punctuation());
    /// assert!(percent.is_ascii_punctuation());
    /// assert!(!space.is_ascii_punctuation());
    /// assert!(!lf.is_ascii_punctuation());
    /// assert!(!esc.is_ascii_punctuation());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_punctuation(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII graphic character:
    /// U+0021 '@' ... U+007E '~'.
    /// For strings, true if all characters in the string are
    /// ASCII punctuation.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(A.is_ascii_graphic());
    /// assert!(G.is_ascii_graphic());
    /// assert!(a.is_ascii_graphic());
    /// assert!(g.is_ascii_graphic());
    /// assert!(zero.is_ascii_graphic());
    /// assert!(percent.is_ascii_graphic());
    /// assert!(!space.is_ascii_graphic());
    /// assert!(!lf.is_ascii_graphic());
    /// assert!(!esc.is_ascii_graphic());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_graphic(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII whitespace character:
    /// U+0020 SPACE, U+0009 HORIZONTAL TAB, U+000A LINE FEED,
    /// U+000C FORM FEED, or U+000D CARRIAGE RETURN.
    /// For strings, true if all characters in the string are
    /// ASCII whitespace.
    ///
    /// Rust uses the WhatWG Infra Standard's [definition of ASCII
    /// whitespace][infra-aw].  There are several other definitions in
    /// wide use.  For instance, [the POSIX locale][pct] includes
    /// U+000B VERTICAL TAB as well as all the above characters,
    /// but—from the very same specification—[the default rule for
    /// "field splitting" in the Bourne shell][bfs] considers *only*
    /// SPACE, HORIZONTAL TAB, and LINE FEED as whitespace.
    ///
    /// If you are writing a program that will process an existing
    /// file format, check what that format's definition of whitespace is
    /// before using this function.
    ///
    /// [infra-aw]: https://infra.spec.whatwg.org/#ascii-whitespace
    /// [pct]: http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_03_01
    /// [bfs]: http://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_05
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(!A.is_ascii_whitespace());
    /// assert!(!G.is_ascii_whitespace());
    /// assert!(!a.is_ascii_whitespace());
    /// assert!(!g.is_ascii_whitespace());
    /// assert!(!zero.is_ascii_whitespace());
    /// assert!(!percent.is_ascii_whitespace());
    /// assert!(space.is_ascii_whitespace());
    /// assert!(lf.is_ascii_whitespace());
    /// assert!(!esc.is_ascii_whitespace());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_whitespace(&self) -> bool { unimplemented!(); }

    /// Checks if the value is an ASCII control character:
    /// U+0000 NUL ... U+001F UNIT SEPARATOR, or U+007F DELETE.
    /// Note that most ASCII whitespace characters are control
    /// characters, but SPACE is not.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    /// # #![allow(non_snake_case)]
    /// use std::ascii::AsciiExt;
    /// let A = 'A';
    /// let G = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc = '\u{001b}';
    ///
    /// assert!(!A.is_ascii_control());
    /// assert!(!G.is_ascii_control());
    /// assert!(!a.is_ascii_control());
    /// assert!(!g.is_ascii_control());
    /// assert!(!zero.is_ascii_control());
    /// assert!(!percent.is_ascii_control());
    /// assert!(!space.is_ascii_control());
    /// assert!(lf.is_ascii_control());
    /// assert!(esc.is_ascii_control());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    fn is_ascii_control(&self) -> bool { unimplemented!(); }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsciiExt for str {
    type Owned = String;

    #[inline]
    fn is_ascii(&self) -> bool {
        self.bytes().all(|b| b.is_ascii())
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> String {
        let mut bytes = self.as_bytes().to_vec();
        bytes.make_ascii_uppercase();
        // make_ascii_uppercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(bytes) }
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> String {
        let mut bytes = self.as_bytes().to_vec();
        bytes.make_ascii_lowercase();
        // make_ascii_uppercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(bytes) }
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    }

    fn make_ascii_uppercase(&mut self) {
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_uppercase()
    }

    fn make_ascii_lowercase(&mut self) {
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_lowercase()
    }

    #[inline]
    fn is_ascii_alphabetic(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_alphabetic())
    }

    #[inline]
    fn is_ascii_uppercase(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_uppercase())
    }

    #[inline]
    fn is_ascii_lowercase(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_lowercase())
    }

    #[inline]
    fn is_ascii_alphanumeric(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_alphanumeric())
    }

    #[inline]
    fn is_ascii_digit(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_digit())
    }

    #[inline]
    fn is_ascii_hexdigit(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_hexdigit())
    }

    #[inline]
    fn is_ascii_punctuation(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_punctuation())
    }

    #[inline]
    fn is_ascii_graphic(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_graphic())
    }

    #[inline]
    fn is_ascii_whitespace(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_whitespace())
    }

    #[inline]
    fn is_ascii_control(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_control())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsciiExt for [u8] {
    type Owned = Vec<u8>;
    #[inline]
    fn is_ascii(&self) -> bool {
        self.iter().all(|b| b.is_ascii())
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> Vec<u8> {
        let mut me = self.to_vec();
        me.make_ascii_uppercase();
        return me
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> Vec<u8> {
        let mut me = self.to_vec();
        me.make_ascii_lowercase();
        return me
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        self.len() == other.len() &&
        self.iter().zip(other).all(|(a, b)| {
            a.eq_ignore_ascii_case(b)
        })
    }

    fn make_ascii_uppercase(&mut self) {
        for byte in self {
            byte.make_ascii_uppercase();
        }
    }

    fn make_ascii_lowercase(&mut self) {
        for byte in self {
            byte.make_ascii_lowercase();
        }
    }

    #[inline]
    fn is_ascii_alphabetic(&self) -> bool {
        self.iter().all(|b| b.is_ascii_alphabetic())
    }

    #[inline]
    fn is_ascii_uppercase(&self) -> bool {
        self.iter().all(|b| b.is_ascii_uppercase())
    }

    #[inline]
    fn is_ascii_lowercase(&self) -> bool {
        self.iter().all(|b| b.is_ascii_lowercase())
    }

    #[inline]
    fn is_ascii_alphanumeric(&self) -> bool {
        self.iter().all(|b| b.is_ascii_alphanumeric())
    }

    #[inline]
    fn is_ascii_digit(&self) -> bool {
        self.iter().all(|b| b.is_ascii_digit())
    }

    #[inline]
    fn is_ascii_hexdigit(&self) -> bool {
        self.iter().all(|b| b.is_ascii_hexdigit())
    }

    #[inline]
    fn is_ascii_punctuation(&self) -> bool {
        self.iter().all(|b| b.is_ascii_punctuation())
    }

    #[inline]
    fn is_ascii_graphic(&self) -> bool {
        self.iter().all(|b| b.is_ascii_graphic())
    }

    #[inline]
    fn is_ascii_whitespace(&self) -> bool {
        self.iter().all(|b| b.is_ascii_whitespace())
    }

    #[inline]
    fn is_ascii_control(&self) -> bool {
        self.iter().all(|b| b.is_ascii_control())
    }
}

macro_rules! impl_by_delegating {
    ($ty:ty, $owned:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl AsciiExt for $ty {
            type Owned = $owned;

            #[inline]
            fn is_ascii(&self) -> bool { self.is_ascii() }

            #[inline]
            fn to_ascii_uppercase(&self) -> Self::Owned { self.to_ascii_uppercase() }

            #[inline]
            fn to_ascii_lowercase(&self) -> Self::Owned { self.to_ascii_lowercase() }

            #[inline]
            fn eq_ignore_ascii_case(&self, o: &Self) -> bool { self.eq_ignore_ascii_case(o) }

            #[inline]
            fn make_ascii_uppercase(&mut self) { self.make_ascii_uppercase(); }

            #[inline]
            fn make_ascii_lowercase(&mut self) { self.make_ascii_lowercase(); }

            #[inline]
            fn is_ascii_alphabetic(&self) -> bool { self.is_ascii_alphabetic() }

            #[inline]
            fn is_ascii_uppercase(&self) -> bool { self.is_ascii_uppercase() }

            #[inline]
            fn is_ascii_lowercase(&self) -> bool { self.is_ascii_lowercase() }

            #[inline]
            fn is_ascii_alphanumeric(&self) -> bool { self.is_ascii_alphanumeric() }

            #[inline]
            fn is_ascii_digit(&self) -> bool { self.is_ascii_digit() }

            #[inline]
            fn is_ascii_hexdigit(&self) -> bool { self.is_ascii_hexdigit() }

            #[inline]
            fn is_ascii_punctuation(&self) -> bool { self.is_ascii_punctuation() }

            #[inline]
            fn is_ascii_graphic(&self) -> bool { self.is_ascii_graphic() }

            #[inline]
            fn is_ascii_whitespace(&self) -> bool { self.is_ascii_whitespace() }

            #[inline]
            fn is_ascii_control(&self) -> bool { self.is_ascii_control() }
        }
    }
}

impl_by_delegating!(u8, u8);
impl_by_delegating!(char, char);

/// An iterator over the escaped version of a byte.
///
/// This `struct` is created by the [`escape_default`] function. See its
/// documentation for more.
///
/// [`escape_default`]: fn.escape_default.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct EscapeDefault {
    range: Range<usize>,
    data: [u8; 4],
}

/// Returns an iterator that produces an escaped version of a `u8`.
///
/// The default is chosen with a bias toward producing literals that are
/// legal in a variety of languages, including C++11 and similar C-family
/// languages. The exact rules are:
///
/// - Tab, CR and LF are escaped as '\t', '\r' and '\n' respectively.
/// - Single-quote, double-quote and backslash chars are backslash-escaped.
/// - Any other chars in the range [0x20,0x7e] are not escaped.
/// - Any other chars are given hex escapes of the form '\xNN'.
/// - Unicode escapes are never generated by this function.
///
/// # Examples
///
/// ```
/// use std::ascii;
///
/// let escaped = ascii::escape_default(b'0').next().unwrap();
/// assert_eq!(b'0', escaped);
///
/// let mut escaped = ascii::escape_default(b'\t');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b't', escaped.next().unwrap());
///
/// let mut escaped = ascii::escape_default(b'\r');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b'r', escaped.next().unwrap());
///
/// let mut escaped = ascii::escape_default(b'\n');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b'n', escaped.next().unwrap());
///
/// let mut escaped = ascii::escape_default(b'\'');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b'\'', escaped.next().unwrap());
///
/// let mut escaped = ascii::escape_default(b'"');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b'"', escaped.next().unwrap());
///
/// let mut escaped = ascii::escape_default(b'\\');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b'\\', escaped.next().unwrap());
///
/// let mut escaped = ascii::escape_default(b'\x9d');
///
/// assert_eq!(b'\\', escaped.next().unwrap());
/// assert_eq!(b'x', escaped.next().unwrap());
/// assert_eq!(b'9', escaped.next().unwrap());
/// assert_eq!(b'd', escaped.next().unwrap());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn escape_default(c: u8) -> EscapeDefault {
    let (data, len) = match c {
        b'\t' => ([b'\\', b't', 0, 0], 2),
        b'\r' => ([b'\\', b'r', 0, 0], 2),
        b'\n' => ([b'\\', b'n', 0, 0], 2),
        b'\\' => ([b'\\', b'\\', 0, 0], 2),
        b'\'' => ([b'\\', b'\'', 0, 0], 2),
        b'"' => ([b'\\', b'"', 0, 0], 2),
        b'\x20' ... b'\x7e' => ([c, 0, 0, 0], 1),
        _ => ([b'\\', b'x', hexify(c >> 4), hexify(c & 0xf)], 4),
    };

    return EscapeDefault { range: (0.. len), data: data };

    fn hexify(b: u8) -> u8 {
        match b {
            0 ... 9 => b'0' + b,
            _ => b'a' + b - 10,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EscapeDefault {
    type Item = u8;
    fn next(&mut self) -> Option<u8> { self.range.next().map(|i| self.data[i]) }
    fn size_hint(&self) -> (usize, Option<usize>) { self.range.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl DoubleEndedIterator for EscapeDefault {
    fn next_back(&mut self) -> Option<u8> {
        self.range.next_back().map(|i| self.data[i])
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ExactSizeIterator for EscapeDefault {}
#[unstable(feature = "fused", issue = "35602")]
impl FusedIterator for EscapeDefault {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for EscapeDefault {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("EscapeDefault { .. }")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use char::from_u32;

    #[test]
    fn test_is_ascii() {
        assert!(b"".is_ascii());
        assert!(b"banana\0\x7F".is_ascii());
        assert!(b"banana\0\x7F".iter().all(|b| b.is_ascii()));
        assert!(!b"Vi\xe1\xbb\x87t Nam".is_ascii());
        assert!(!b"Vi\xe1\xbb\x87t Nam".iter().all(|b| b.is_ascii()));
        assert!(!b"\xe1\xbb\x87".iter().any(|b| b.is_ascii()));

        assert!("".is_ascii());
        assert!("banana\0\u{7F}".is_ascii());
        assert!("banana\0\u{7F}".chars().all(|c| c.is_ascii()));
        assert!(!"ประเทศไทย中华Việt Nam".chars().all(|c| c.is_ascii()));
        assert!(!"ประเทศไทย中华ệ ".chars().any(|c| c.is_ascii()));
    }

    #[test]
    fn test_to_ascii_uppercase() {
        assert_eq!("url()URL()uRl()ürl".to_ascii_uppercase(), "URL()URL()URL()üRL");
        assert_eq!("hıKß".to_ascii_uppercase(), "HıKß");

        for i in 0..501 {
            let upper = if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().to_ascii_uppercase(),
                       (from_u32(upper).unwrap()).to_string());
        }
    }

    #[test]
    fn test_to_ascii_lowercase() {
        assert_eq!("url()URL()uRl()Ürl".to_ascii_lowercase(), "url()url()url()Ürl");
        // Dotted capital I, Kelvin sign, Sharp S.
        assert_eq!("HİKß".to_ascii_lowercase(), "hİKß");

        for i in 0..501 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().to_ascii_lowercase(),
                       (from_u32(lower).unwrap()).to_string());
        }
    }

    #[test]
    fn test_make_ascii_lower_case() {
        macro_rules! test {
            ($from: expr, $to: expr) => {
                {
                    let mut x = $from;
                    x.make_ascii_lowercase();
                    assert_eq!(x, $to);
                }
            }
        }
        test!(b'A', b'a');
        test!(b'a', b'a');
        test!(b'!', b'!');
        test!('A', 'a');
        test!('À', 'À');
        test!('a', 'a');
        test!('!', '!');
        test!(b"H\xc3\x89".to_vec(), b"h\xc3\x89");
        test!("HİKß".to_string(), "hİKß");
    }


    #[test]
    fn test_make_ascii_upper_case() {
        macro_rules! test {
            ($from: expr, $to: expr) => {
                {
                    let mut x = $from;
                    x.make_ascii_uppercase();
                    assert_eq!(x, $to);
                }
            }
        }
        test!(b'a', b'A');
        test!(b'A', b'A');
        test!(b'!', b'!');
        test!('a', 'A');
        test!('à', 'à');
        test!('A', 'A');
        test!('!', '!');
        test!(b"h\xc3\xa9".to_vec(), b"H\xc3\xa9");
        test!("hıKß".to_string(), "HıKß");

        let mut x = "Hello".to_string();
        x[..3].make_ascii_uppercase();  // Test IndexMut on String.
        assert_eq!(x, "HELlo")
    }

    #[test]
    fn test_eq_ignore_ascii_case() {
        assert!("url()URL()uRl()Ürl".eq_ignore_ascii_case("url()url()url()Ürl"));
        assert!(!"Ürl".eq_ignore_ascii_case("ürl"));
        // Dotted capital I, Kelvin sign, Sharp S.
        assert!("HİKß".eq_ignore_ascii_case("hİKß"));
        assert!(!"İ".eq_ignore_ascii_case("i"));
        assert!(!"K".eq_ignore_ascii_case("k"));
        assert!(!"ß".eq_ignore_ascii_case("s"));

        for i in 0..501 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert!((from_u32(i).unwrap()).to_string().eq_ignore_ascii_case(
                    &from_u32(lower).unwrap().to_string()));
        }
    }

    #[test]
    fn inference_works() {
        let x = "a".to_string();
        x.eq_ignore_ascii_case("A");
    }

    // Shorthands used by the is_ascii_* tests.
    macro_rules! assert_all {
        ($what:ident, $($str:tt),+) => {{
            $(
                for b in $str.chars() {
                    if !b.$what() {
                        panic!("expected {}({}) but it isn't",
                               stringify!($what), b);
                    }
                }
                for b in $str.as_bytes().iter() {
                    if !b.$what() {
                        panic!("expected {}(0x{:02x})) but it isn't",
                               stringify!($what), b);
                    }
                }
                assert!($str.$what());
                assert!($str.as_bytes().$what());
            )+
        }};
        ($what:ident, $($str:tt),+,) => (assert_all!($what,$($str),+))
    }
    macro_rules! assert_none {
        ($what:ident, $($str:tt),+) => {{
            $(
                for b in $str.chars() {
                    if b.$what() {
                        panic!("expected not-{}({}) but it is",
                               stringify!($what), b);
                    }
                }
                for b in $str.as_bytes().iter() {
                    if b.$what() {
                        panic!("expected not-{}(0x{:02x})) but it is",
                               stringify!($what), b);
                    }
                }
            )*
        }};
        ($what:ident, $($str:tt),+,) => (assert_none!($what,$($str),+))
    }

    #[test]
    fn test_is_ascii_alphabetic() {
        assert_all!(is_ascii_alphabetic,
            "",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        );
        assert_none!(is_ascii_alphabetic,
            "0123456789",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_uppercase() {
        assert_all!(is_ascii_uppercase,
            "",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
        );
        assert_none!(is_ascii_uppercase,
            "abcdefghijklmnopqrstuvwxyz",
            "0123456789",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_lowercase() {
        assert_all!(is_ascii_lowercase,
            "abcdefghijklmnopqrstuvwxyz",
        );
        assert_none!(is_ascii_lowercase,
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "0123456789",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_alphanumeric() {
        assert_all!(is_ascii_alphanumeric,
            "",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "0123456789",
        );
        assert_none!(is_ascii_alphanumeric,
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_digit() {
        assert_all!(is_ascii_digit,
            "",
            "0123456789",
        );
        assert_none!(is_ascii_digit,
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_hexdigit() {
        assert_all!(is_ascii_hexdigit,
            "",
            "0123456789",
            "abcdefABCDEF",
        );
        assert_none!(is_ascii_hexdigit,
            "ghijklmnopqrstuvwxyz",
            "GHIJKLMNOQPRSTUVWXYZ",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_punctuation() {
        assert_all!(is_ascii_punctuation,
            "",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        );
        assert_none!(is_ascii_punctuation,
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "0123456789",
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_graphic() {
        assert_all!(is_ascii_graphic,
            "",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "0123456789",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        );
        assert_none!(is_ascii_graphic,
            " \t\n\x0c\r",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_whitespace() {
        assert_all!(is_ascii_whitespace,
            "",
            " \t\n\x0c\r",
        );
        assert_none!(is_ascii_whitespace,
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "0123456789",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x0b\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
    }

    #[test]
    fn test_is_ascii_control() {
        assert_all!(is_ascii_control,
            "",
            "\x00\x01\x02\x03\x04\x05\x06\x07",
            "\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f",
            "\x10\x11\x12\x13\x14\x15\x16\x17",
            "\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f",
            "\x7f",
        );
        assert_none!(is_ascii_control,
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOQPRSTUVWXYZ",
            "0123456789",
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            " ",
        );
    }
}
