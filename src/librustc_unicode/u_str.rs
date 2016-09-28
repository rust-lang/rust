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

use core::char;
use core::iter::{Filter, FusedIterator};
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
    fn trim(&self) -> &str;
    fn trim_left(&self) -> &str;
    fn trim_right(&self) -> &str;
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

impl<I> Iterator for Utf16Encoder<I>
    where I: Iterator<Item = char>
{
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
            let n = CharExt::encode_utf16(ch, &mut buf).len();
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

#[unstable(feature = "fused", issue = "35602")]
impl<I> FusedIterator for Utf16Encoder<I>
    where I: FusedIterator<Item = char> {}

#[stable(feature = "split_whitespace", since = "1.1.0")]
impl<'a> Iterator for SplitWhitespace<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        self.inner.next()
    }
}

#[stable(feature = "split_whitespace", since = "1.1.0")]
impl<'a> DoubleEndedIterator for SplitWhitespace<'a> {
    fn next_back(&mut self) -> Option<&'a str> {
        self.inner.next_back()
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a> FusedIterator for SplitWhitespace<'a> {}
