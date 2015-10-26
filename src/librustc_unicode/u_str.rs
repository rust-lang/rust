// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unicode-intensive string manipulations.
//!
//! This module provides functionality to `str` that requires the Unicode
//! methods provided by the unicode parts of the CharExt trait.

use char::{DecodeUtf16, decode_utf16};
use core::char;
use core::iter::{Cloned, Filter};
use core::slice;
use core::str::Split;

/// An iterator over the non-whitespace substrings of a string,
/// separated by any amount of whitespace.
#[stable(feature = "split_whitespace", since = "1.1.0")]
pub struct SplitWhitespace<'a> {
    inner: Filter<Split<'a, fn(char) -> bool>, fn(&&str) -> bool>,
}

/// Methods for Unicode string slices
#[allow(missing_docs)] // docs in libcollections
pub trait UnicodeStr {
    fn split_whitespace<'a>(&'a self) -> SplitWhitespace<'a>;
    fn is_whitespace(&self) -> bool;
    fn is_alphanumeric(&self) -> bool;
    fn trim<'a>(&'a self) -> &'a str;
    fn trim_left<'a>(&'a self) -> &'a str;
    fn trim_right<'a>(&'a self) -> &'a str;
}

impl UnicodeStr for str {
    #[inline]
    fn split_whitespace(&self) -> SplitWhitespace {
        fn is_not_empty(s: &&str) -> bool {
            !s.is_empty()
        }
        let is_not_empty: fn(&&str) -> bool = is_not_empty; // coerce to fn pointer

        fn is_whitespace(c: char) -> bool {
            c.is_whitespace()
        }
        let is_whitespace: fn(char) -> bool = is_whitespace; // coerce to fn pointer

        SplitWhitespace { inner: self.split(is_whitespace).filter(is_not_empty) }
    }

    #[inline]
    fn is_whitespace(&self) -> bool {
        self.chars().all(|c| c.is_whitespace())
    }

    #[inline]
    fn is_alphanumeric(&self) -> bool {
        self.chars().all(|c| c.is_alphanumeric())
    }

    #[inline]
    fn trim(&self) -> &str {
        self.trim_matches(|c: char| c.is_whitespace())
    }

    #[inline]
    fn trim_left(&self) -> &str {
        self.trim_left_matches(|c: char| c.is_whitespace())
    }

    #[inline]
    fn trim_right(&self) -> &str {
        self.trim_right_matches(|c: char| c.is_whitespace())
    }
}

// https://tools.ietf.org/html/rfc3629
static UTF8_CHAR_WIDTH: [u8; 256] = [
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x1F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x3F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x5F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x7F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0x9F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0xBF
0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // 0xDF
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, // 0xEF
4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0, // 0xFF
];

/// Given a first byte, determine how many bytes are in this UTF-8 character
#[inline]
pub fn utf8_char_width(b: u8) -> usize {
    return UTF8_CHAR_WIDTH[b as usize] as usize;
}

/// Determines if a vector of `u16` contains valid UTF-16
pub fn is_utf16(v: &[u16]) -> bool {
    let mut it = v.iter();
    macro_rules! next { ($ret:expr) => {
            match it.next() { Some(u) => *u, None => return $ret }
        }
    }
    loop {
        let u = next!(true);

        match char::from_u32(u as u32) {
            Some(_) => {}
            None => {
                let u2 = next!(false);
                if u < 0xD7FF || u > 0xDBFF || u2 < 0xDC00 || u2 > 0xDFFF {
                    return false;
                }
            }
        }
    }
}

/// An iterator that decodes UTF-16 encoded codepoints from a vector
/// of `u16`s.
#[deprecated(since = "1.4.0", reason = "renamed to `char::DecodeUtf16`")]
#[unstable(feature = "decode_utf16", reason = "not exposed in std", issue = "27830")]
#[allow(deprecated)]
#[derive(Clone)]
pub struct Utf16Items<'a> {
    decoder: DecodeUtf16<Cloned<slice::Iter<'a, u16>>>,
}

/// The possibilities for values decoded from a `u16` stream.
#[deprecated(since = "1.4.0", reason = "`char::DecodeUtf16` uses `Result<char, u16>` instead")]
#[unstable(feature = "decode_utf16", reason = "not exposed in std", issue = "27830")]
#[allow(deprecated)]
#[derive(Copy, PartialEq, Eq, Clone, Debug)]
pub enum Utf16Item {
    /// A valid codepoint.
    ScalarValue(char),
    /// An invalid surrogate without its pair.
    LoneSurrogate(u16),
}

#[allow(deprecated)]
impl Utf16Item {
    /// Convert `self` to a `char`, taking `LoneSurrogate`s to the
    /// replacement character (U+FFFD).
    #[inline]
    pub fn to_char_lossy(&self) -> char {
        match *self {
            Utf16Item::ScalarValue(c) => c,
            Utf16Item::LoneSurrogate(_) => '\u{FFFD}',
        }
    }
}

#[deprecated(since = "1.4.0", reason = "use `char::DecodeUtf16` instead")]
#[unstable(feature = "decode_utf16", reason = "not exposed in std", issue = "27830")]
#[allow(deprecated)]
impl<'a> Iterator for Utf16Items<'a> {
    type Item = Utf16Item;

    fn next(&mut self) -> Option<Utf16Item> {
        self.decoder.next().map(|result| {
            match result {
                Ok(c) => Utf16Item::ScalarValue(c),
                Err(s) => Utf16Item::LoneSurrogate(s),
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.decoder.size_hint()
    }
}

/// Create an iterator over the UTF-16 encoded codepoints in `v`,
/// returning invalid surrogates as `LoneSurrogate`s.
///
/// # Examples
///
/// ```
/// #![feature(unicode, decode_utf16)]
///
/// extern crate rustc_unicode;
///
/// use rustc_unicode::str::Utf16Item::{ScalarValue, LoneSurrogate};
///
/// fn main() {
///     // 𝄞mus<invalid>ic<invalid>
///     let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///              0x0073, 0xDD1E, 0x0069, 0x0063,
///              0xD834];
///
///     assert_eq!(rustc_unicode::str::utf16_items(&v).collect::<Vec<_>>(),
///                vec![ScalarValue('𝄞'),
///                     ScalarValue('m'), ScalarValue('u'), ScalarValue('s'),
///                     LoneSurrogate(0xDD1E),
///                     ScalarValue('i'), ScalarValue('c'),
///                     LoneSurrogate(0xD834)]);
/// }
/// ```
#[deprecated(since = "1.4.0", reason = "renamed to `char::decode_utf16`")]
#[unstable(feature = "decode_utf16", reason = "not exposed in std", issue = "27830")]
#[allow(deprecated)]
pub fn utf16_items<'a>(v: &'a [u16]) -> Utf16Items<'a> {
    Utf16Items { decoder: decode_utf16(v.iter().cloned()) }
}

/// Iterator adaptor for encoding `char`s to UTF-16.
#[derive(Clone)]
pub struct Utf16Encoder<I> {
    chars: I,
    extra: u16,
}

impl<I> Utf16Encoder<I> {
    /// Create a UTF-16 encoder from any `char` iterator.
    pub fn new(chars: I) -> Utf16Encoder<I>
        where I: Iterator<Item = char>
    {
        Utf16Encoder {
            chars: chars,
            extra: 0,
        }
    }
}

impl<I> Iterator for Utf16Encoder<I> where I: Iterator<Item=char> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0; 2];
        self.chars.next().map(|ch| {
            let n = CharExt::encode_utf16(ch, &mut buf).unwrap_or(0);
            if n == 2 {
                self.extra = buf[1];
            }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.chars.size_hint();
        // every char gets either one u16 or two u16,
        // so this iterator is between 1 or 2 times as
        // long as the underlying iterator.
        (low, high.and_then(|n| n.checked_mul(2)))
    }
}

impl<'a> Iterator for SplitWhitespace<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        self.inner.next()
    }
}
impl<'a> DoubleEndedIterator for SplitWhitespace<'a> {
    fn next_back(&mut self) -> Option<&'a str> {
        self.inner.next_back()
    }
}
