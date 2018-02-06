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
///
/// This struct is created by the [`split_whitespace`] method on [`str`].
/// See its documentation for more.
///
/// [`split_whitespace`]: ../../std/primitive.str.html#method.split_whitespace
/// [`str`]: ../../std/primitive.str.html
#[stable(feature = "split_whitespace", since = "1.1.0")]
#[derive(Clone, Debug)]
pub struct SplitWhitespace<'a> {
    inner: Filter<Split<'a, IsWhitespace>, IsNotEmpty>,
}

/// Methods for Unicode string slices
#[allow(missing_docs)] // docs in liballoc
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
        SplitWhitespace { inner: self.split(IsWhitespace).filter(IsNotEmpty) }
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

/// Iterator adaptor for encoding `char`s to UTF-16.
#[derive(Clone)]
#[allow(missing_debug_implementations)]
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
            chars,
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

#[derive(Clone)]
struct IsWhitespace;

impl FnOnce<(char, )> for IsWhitespace {
    type Output = bool;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (char, )) -> bool {
        self.call_mut(arg)
    }
}

impl FnMut<(char, )> for IsWhitespace {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (char, )) -> bool {
        arg.0.is_whitespace()
    }
}

#[derive(Clone)]
struct IsNotEmpty;

impl<'a, 'b> FnOnce<(&'a &'b str, )> for IsNotEmpty {
    type Output = bool;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (&&str, )) -> bool {
        self.call_mut(arg)
    }
}

impl<'a, 'b> FnMut<(&'a &'b str, )> for IsNotEmpty {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (&&str, )) -> bool {
        !arg.0.is_empty()
    }
}


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
