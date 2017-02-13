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

#![stable(feature = "rust1", since = "1.0.0")]

use fmt;
use mem;
use ops::Range;
use iter::FusedIterator;

/// Extension methods for ASCII-subset only operations on string slices.
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
    /// let utf8 = '❤';
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!utf8.is_ascii());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_ascii(&self) -> bool;

    /// Makes a copy of the string in ASCII upper case.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the string in-place, use [`make_ascii_uppercase`].
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
    /// let utf8 = '❤';
    ///
    /// assert_eq!('A', ascii.to_ascii_uppercase());
    /// assert_eq!('❤', utf8.to_ascii_uppercase());
    /// ```
    ///
    /// [`make_ascii_uppercase`]: #tymethod.make_ascii_uppercase
    /// [`str::to_uppercase`]: ../primitive.str.html#method.to_uppercase
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_ascii_uppercase(&self) -> Self::Owned;

    /// Makes a copy of the string in ASCII lower case.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the string in-place, use [`make_ascii_lowercase`].
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
    /// let utf8 = '❤';
    ///
    /// assert_eq!('a', ascii.to_ascii_lowercase());
    /// assert_eq!('❤', utf8.to_ascii_lowercase());
    /// ```
    ///
    /// [`make_ascii_lowercase`]: #tymethod.make_ascii_lowercase
    /// [`str::to_lowercase`]: ../primitive.str.html#method.to_lowercase
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_ascii_lowercase(&self) -> Self::Owned;

    /// Checks that two strings are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporary strings.
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
    /// To return a new uppercased string without modifying the existing one, use
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
    /// To return a new lowercased string without modifying the existing one, use
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
    /// U+0021 ... U+002F `! " # $ % & ' ( ) * + , - . /`
    /// U+003A ... U+0040 `: ; < = > ? @`
    /// U+005B ... U+0060 `[ \\ ] ^ _ \``
    /// U+007B ... U+007E `{ | } ~`
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
        let me: &mut [u8] = unsafe { mem::transmute(self) };
        me.make_ascii_uppercase()
    }

    fn make_ascii_lowercase(&mut self) {
        let me: &mut [u8] = unsafe { mem::transmute(self) };
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

#[stable(feature = "rust1", since = "1.0.0")]
impl AsciiExt for u8 {
    type Owned = u8;
    #[inline]
    fn is_ascii(&self) -> bool { *self & 128 == 0 }
    #[inline]
    fn to_ascii_uppercase(&self) -> u8 { ASCII_UPPERCASE_MAP[*self as usize] }
    #[inline]
    fn to_ascii_lowercase(&self) -> u8 { ASCII_LOWERCASE_MAP[*self as usize] }
    #[inline]
    fn eq_ignore_ascii_case(&self, other: &u8) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }
    #[inline]
    fn make_ascii_uppercase(&mut self) { *self = self.to_ascii_uppercase(); }
    #[inline]
    fn make_ascii_lowercase(&mut self) { *self = self.to_ascii_lowercase(); }

    #[inline]
    fn is_ascii_alphabetic(&self) -> bool {
        if *self >= 0x80 { return false; }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            L|Lx|U|Ux => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_uppercase(&self) -> bool {
        if *self >= 0x80 { return false }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            U|Ux => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_lowercase(&self) -> bool {
        if *self >= 0x80 { return false }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            L|Lx => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_alphanumeric(&self) -> bool {
        if *self >= 0x80 { return false }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            D|L|Lx|U|Ux => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_digit(&self) -> bool {
        if *self >= 0x80 { return false }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            D => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_hexdigit(&self) -> bool {
        if *self >= 0x80 { return false }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            D|Lx|Ux => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_punctuation(&self) -> bool {
        if *self >= 0x80 { return false }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            P => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_graphic(&self) -> bool {
        if *self >= 0x80 { return false; }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            Ux|U|Lx|L|D|P => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_whitespace(&self) -> bool {
        if *self >= 0x80 { return false; }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            Cw|W => true,
            _ => false
        }
    }

    #[inline]
    fn is_ascii_control(&self) -> bool {
        if *self >= 0x80 { return false; }
        match ASCII_CHARACTER_CLASS[*self as usize] {
            C|Cw => true,
            _ => false
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsciiExt for char {
    type Owned = char;
    #[inline]
    fn is_ascii(&self) -> bool {
        *self as u32 <= 0x7F
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> char {
        if self.is_ascii() {
            (*self as u8).to_ascii_uppercase() as char
        } else {
            *self
        }
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> char {
        if self.is_ascii() {
            (*self as u8).to_ascii_lowercase() as char
        } else {
            *self
        }
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &char) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }

    #[inline]
    fn make_ascii_uppercase(&mut self) { *self = self.to_ascii_uppercase(); }
    #[inline]
    fn make_ascii_lowercase(&mut self) { *self = self.to_ascii_lowercase(); }

    #[inline]
    fn is_ascii_alphabetic(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_alphabetic()
    }

    #[inline]
    fn is_ascii_uppercase(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_uppercase()
    }

    #[inline]
    fn is_ascii_lowercase(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_lowercase()
    }

    #[inline]
    fn is_ascii_alphanumeric(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_alphanumeric()
    }

    #[inline]
    fn is_ascii_digit(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_digit()
    }

    #[inline]
    fn is_ascii_hexdigit(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_hexdigit()
    }

    #[inline]
    fn is_ascii_punctuation(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_punctuation()
    }

    #[inline]
    fn is_ascii_graphic(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_graphic()
    }

    #[inline]
    fn is_ascii_whitespace(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_whitespace()
    }

    #[inline]
    fn is_ascii_control(&self) -> bool {
        (*self as u32 <= 0x7f) && (*self as u8).is_ascii_control()
    }
}

/// An iterator over the escaped version of a byte, constructed via
/// `std::ascii::escape_default`.
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


static ASCII_LOWERCASE_MAP: [u8; 256] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    b' ', b'!', b'"', b'#', b'$', b'%', b'&', b'\'',
    b'(', b')', b'*', b'+', b',', b'-', b'.', b'/',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b':', b';', b'<', b'=', b'>', b'?',
    b'@',

          b'a', b'b', b'c', b'd', b'e', b'f', b'g',
    b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o',
    b'p', b'q', b'r', b's', b't', b'u', b'v', b'w',
    b'x', b'y', b'z',

                      b'[', b'\\', b']', b'^', b'_',
    b'`', b'a', b'b', b'c', b'd', b'e', b'f', b'g',
    b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o',
    b'p', b'q', b'r', b's', b't', b'u', b'v', b'w',
    b'x', b'y', b'z', b'{', b'|', b'}', b'~', 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

static ASCII_UPPERCASE_MAP: [u8; 256] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    b' ', b'!', b'"', b'#', b'$', b'%', b'&', b'\'',
    b'(', b')', b'*', b'+', b',', b'-', b'.', b'/',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b':', b';', b'<', b'=', b'>', b'?',
    b'@', b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
    b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W',
    b'X', b'Y', b'Z', b'[', b'\\', b']', b'^', b'_',
    b'`',

          b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
    b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W',
    b'X', b'Y', b'Z',

                      b'{', b'|', b'}', b'~', 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

enum AsciiCharacterClass {
    C,  // control
    Cw, // control whitespace
    W,  // whitespace
    D,  // digit
    L,  // lowercase
    Lx, // lowercase hex digit
    U,  // uppercase
    Ux, // uppercase hex digit
    P,  // punctuation
}
use self::AsciiCharacterClass::*;

static ASCII_CHARACTER_CLASS: [AsciiCharacterClass; 128] = [
//  _0 _1 _2 _3 _4 _5 _6 _7 _8 _9 _a _b _c _d _e _f
    C, C, C, C, C, C, C, C, C, Cw,Cw,C, Cw,Cw,C, C, // 0_
    C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, // 1_
    W, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, // 2_
    D, D, D, D, D, D, D, D, D, D, P, P, P, P, P, P, // 3_
    P, Ux,Ux,Ux,Ux,Ux,Ux,U, U, U, U, U, U, U, U, U, // 4_
    U, U, U, U, U, U, U, U, U, U, U, P, P, P, P, P, // 5_
    P, Lx,Lx,Lx,Lx,Lx,Lx,L, L, L, L, L, L, L, L, L, // 6_
    L, L, L, L, L, L, L, L, L, L, L, P, P, P, P, C, // 7_
];

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
