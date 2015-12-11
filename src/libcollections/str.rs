// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unicode string slices
//!
//! *[See also the `str` primitive type](../primitive.str.html).*


#![stable(feature = "rust1", since = "1.0.0")]

// Many of the usings in this module are only used in the test configuration.
// It's cleaner to just turn off the unused_imports warning than to fix them.
#![allow(unused_imports)]

use core::clone::Clone;
use core::iter::{Iterator, Extend};
use core::option::Option::{self, Some, None};
use core::result::Result;
use core::str as core_str;
use core::str::pattern::Pattern;
use core::str::pattern::{Searcher, ReverseSearcher, DoubleEndedSearcher};
use core::mem;
use rustc_unicode::str::{UnicodeStr, Utf16Encoder};

use vec_deque::VecDeque;
use borrow::{Borrow, ToOwned};
use string::String;
use rustc_unicode;
use vec::Vec;
use slice::SliceConcatExt;
use boxed::Box;

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{FromStr, Utf8Error};
#[allow(deprecated)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Lines, LinesAny, CharRange};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Split, RSplit};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{SplitN, RSplitN};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{SplitTerminator, RSplitTerminator};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Matches, RMatches};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{MatchIndices, RMatchIndices};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{from_utf8, Chars, CharIndices, Bytes};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{from_utf8_unchecked, ParseBoolError};
#[stable(feature = "rust1", since = "1.0.0")]
pub use rustc_unicode::str::SplitWhitespace;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::pattern;

#[unstable(feature = "slice_concat_ext",
           reason = "trait should not have to exist",
           issue = "27747")]
impl<S: Borrow<str>> SliceConcatExt<str> for [S] {
    type Output = String;

    fn concat(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        // `len` calculation may overflow but push_str will check boundaries
        let len = self.iter().map(|s| s.borrow().len()).sum();
        let mut result = String::with_capacity(len);

        for s in self {
            result.push_str(s.borrow())
        }

        result
    }

    fn join(&self, sep: &str) -> String {
        if self.is_empty() {
            return String::new();
        }

        // concat is faster
        if sep.is_empty() {
            return self.concat();
        }

        // this is wrong without the guarantee that `self` is non-empty
        // `len` calculation may overflow but push_str but will check boundaries
        let len = sep.len() * (self.len() - 1) +
                  self.iter().map(|s| s.borrow().len()).sum::<usize>();
        let mut result = String::with_capacity(len);
        let mut first = true;

        for s in self {
            if first {
                first = false;
            } else {
                result.push_str(sep);
            }
            result.push_str(s.borrow());
        }
        result
    }

    fn connect(&self, sep: &str) -> String {
        self.join(sep)
    }
}

/// External iterator for a string's UTF-16 code units.
///
/// For use with the `std::iter` module.
#[derive(Clone)]
#[unstable(feature = "str_utf16", issue = "27714")]
pub struct Utf16Units<'a> {
    encoder: Utf16Encoder<Chars<'a>>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Utf16Units<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        self.encoder.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.encoder.size_hint()
    }
}

// Return the initial codepoint accumulator for the first byte.
// The first byte is special, only want bottom 5 bits for width 2, 4 bits
// for width 3, and 3 bits for width 4
macro_rules! utf8_first_byte {
    ($byte:expr, $width:expr) => (($byte & (0x7F >> $width)) as u32)
}

// return the value of $ch updated with continuation byte $byte
macro_rules! utf8_acc_cont_byte {
    ($ch:expr, $byte:expr) => (($ch << 6) | ($byte & 63) as u32)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<str> for String {
    #[inline]
    fn borrow(&self) -> &str {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for str {
    type Owned = String;
    fn to_owned(&self) -> String {
        unsafe { String::from_utf8_unchecked(self.as_bytes().to_owned()) }
    }
}

/// Methods for string slices.
#[lang = "str"]
#[cfg(not(test))]
impl str {
    /// Returns the length of `self`.
    ///
    /// This length is in bytes, not [`char`]s or graphemes. In other words,
    /// it may not be what a human considers the length of the string.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let len = "foo".len();
    /// assert_eq!(3, len);
    ///
    /// let len = "ƒoo".len(); // fancy f!
    /// assert_eq!(4, len);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len(&self) -> usize {
        core_str::StrExt::len(self)
    }

    /// Returns true if this slice has a length of zero bytes.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "";
    /// assert!(s.is_empty());
    ///
    /// let s = "not empty";
    /// assert!(!s.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        core_str::StrExt::is_empty(self)
    }

    /// Checks that `index`-th byte lies at the start and/or end of a
    /// UTF-8 code point sequence.
    ///
    /// The start and end of the string (when `index == self.len()`) are
    /// considered to be
    /// boundaries.
    ///
    /// Returns `false` if `index` is greater than `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_char)]
    ///
    /// let s = "Löwe 老虎 Léopard";
    /// assert!(s.is_char_boundary(0));
    /// // start of `老`
    /// assert!(s.is_char_boundary(6));
    /// assert!(s.is_char_boundary(s.len()));
    ///
    /// // second byte of `ö`
    /// assert!(!s.is_char_boundary(2));
    ///
    /// // third byte of `老`
    /// assert!(!s.is_char_boundary(8));
    /// ```
    #[unstable(feature = "str_char",
               reason = "it is unclear whether this method pulls its weight \
                         with the existence of the char_indices iterator or \
                         this method may want to be replaced with checked \
                         slicing",
               issue = "27754")]
    #[inline]
    pub fn is_char_boundary(&self, index: usize) -> bool {
        core_str::StrExt::is_char_boundary(self, index)
    }

    /// Converts a string slice to a byte slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bytes = "bors".as_bytes();
    /// assert_eq!(b"bors", bytes);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        core_str::StrExt::as_bytes(self)
    }

    /// Converts a string slice to a raw pointer.
    ///
    /// As string slices are a slice of bytes, the raw pointer points to a
    /// `u8`. This pointer will be pointing to the first byte of the string
    /// slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Hello";
    /// let ptr = s.as_ptr();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        core_str::StrExt::as_ptr(self)
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get a mutable string slice instead, see the
    /// [`slice_mut_unchecked()`] method.
    ///
    /// [`slice_mut_unchecked()`]: #method.slice_mut_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisifed:
    ///
    /// * `begin` must come before `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// unsafe {
    ///     assert_eq!("Löwe 老虎 Léopard", s.slice_unchecked(0, 21));
    /// }
    ///
    /// let s = "Hello, world!";
    ///
    /// unsafe {
    ///     assert_eq!("world", s.slice_unchecked(7, 12));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        core_str::StrExt::slice_unchecked(self, begin, end)
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get an immutable string slice instead, see the
    /// [`slice_unchecked()`] method.
    ///
    /// [`slice_unchecked()`]: #method.slice_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisifed:
    ///
    /// * `begin` must come before `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    #[stable(feature = "str_slice_mut", since = "1.5.0")]
    #[inline]
    pub unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str {
        core_str::StrExt::slice_mut_unchecked(self, begin, end)
    }

    /// Given a byte position, returns the next `char` and its index.
    ///
    /// # Panics
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 sequence.
    ///
    /// # Examples
    ///
    /// This example manually iterates through the code points of a string;
    /// this should normally be
    /// done by `.chars()` or `.char_indices()`.
    ///
    /// ```
    /// #![feature(str_char)]
    ///
    /// use std::str::CharRange;
    ///
    /// let s = "中华Việt Nam";
    /// let mut i = 0;
    /// while i < s.len() {
    ///     let CharRange {ch, next} = s.char_range_at(i);
    ///     println!("{}: {}", i, ch);
    ///     i = next;
    /// }
    /// ```
    ///
    /// This outputs:
    ///
    /// ```text
    /// 0: 中
    /// 3: 华
    /// 6: V
    /// 7: i
    /// 8: e
    /// 9:
    /// 11:
    /// 13: t
    /// 14:
    /// 15: N
    /// 16: a
    /// 17: m
    /// ```
    #[unstable(feature = "str_char",
               reason = "often replaced by char_indices, this method may \
                         be removed in favor of just char_at() or eventually \
                         removed altogether",
               issue = "27754")]
    #[inline]
    pub fn char_range_at(&self, start: usize) -> CharRange {
        core_str::StrExt::char_range_at(self, start)
    }

    /// Given a byte position, returns the previous `char` and its position.
    ///
    /// Note that Unicode has many features, such as combining marks, ligatures,
    /// and direction marks, that need to be taken into account to correctly reverse a string.
    ///
    /// Returns 0 for next index if called on start index 0.
    ///
    /// # Panics
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 sequence.
    ///
    /// # Examples
    ///
    /// This example manually iterates through the code points of a string;
    /// this should normally be
    /// done by `.chars().rev()` or `.char_indices()`.
    ///
    /// ```
    /// #![feature(str_char)]
    ///
    /// use std::str::CharRange;
    ///
    /// let s = "中华Việt Nam";
    /// let mut i = s.len();
    /// while i > 0 {
    ///     let CharRange {ch, next} = s.char_range_at_reverse(i);
    ///     println!("{}: {}", i, ch);
    ///     i = next;
    /// }
    /// ```
    ///
    /// This outputs:
    ///
    /// ```text
    /// 18: m
    /// 17: a
    /// 16: N
    /// 15:
    /// 14: t
    /// 13:
    /// 11:
    /// 9: e
    /// 8: i
    /// 7: V
    /// 6: 华
    /// 3: 中
    /// ```
    #[unstable(feature = "str_char",
               reason = "often replaced by char_indices, this method may \
                         be removed in favor of just char_at_reverse() or \
                         eventually removed altogether",
               issue = "27754")]
    #[inline]
    pub fn char_range_at_reverse(&self, start: usize) -> CharRange {
        core_str::StrExt::char_range_at_reverse(self, start)
    }

    /// Given a byte position, returns the `char` at that position.
    ///
    /// # Panics
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_char)]
    ///
    /// let s = "abπc";
    /// assert_eq!(s.char_at(1), 'b');
    /// assert_eq!(s.char_at(2), 'π');
    /// assert_eq!(s.char_at(4), 'c');
    /// ```
    #[unstable(feature = "str_char",
               reason = "frequently replaced by the chars() iterator, this \
                         method may be removed or possibly renamed in the \
                         future; it is normally replaced by chars/char_indices \
                         iterators or by getting the first char from a \
                         subslice",
               issue = "27754")]
    #[inline]
    pub fn char_at(&self, i: usize) -> char {
        core_str::StrExt::char_at(self, i)
    }

    /// Given a byte position, returns the `char` at that position, counting
    /// from the end.
    ///
    /// # Panics
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_char)]
    ///
    /// let s = "abπc";
    /// assert_eq!(s.char_at_reverse(1), 'a');
    /// assert_eq!(s.char_at_reverse(2), 'b');
    /// assert_eq!(s.char_at_reverse(3), 'π');
    /// ```
    #[unstable(feature = "str_char",
               reason = "see char_at for more details, but reverse semantics \
                         are also somewhat unclear, especially with which \
                         cases generate panics",
               issue = "27754")]
    #[inline]
    pub fn char_at_reverse(&self, i: usize) -> char {
        core_str::StrExt::char_at_reverse(self, i)
    }

    /// Retrieves the first `char` from a `&str` and returns it.
    ///
    /// Note that a single Unicode character (grapheme cluster)
    /// can be composed of multiple `char`s.
    ///
    /// This does not allocate a new string; instead, it returns a slice that
    /// points one code point beyond the code point that was shifted.
    ///
    /// `None` is returned if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_char)]
    ///
    /// let s = "Łódź"; // \u{141}o\u{301}dz\u{301}
    /// let (c, s1) = s.slice_shift_char().unwrap();
    ///
    /// assert_eq!(c, 'Ł');
    /// assert_eq!(s1, "ódź");
    ///
    /// let (c, s2) = s1.slice_shift_char().unwrap();
    ///
    /// assert_eq!(c, 'o');
    /// assert_eq!(s2, "\u{301}dz\u{301}");
    /// ```
    #[unstable(feature = "str_char",
               reason = "awaiting conventions about shifting and slices and \
                         may not be warranted with the existence of the chars \
                         and/or char_indices iterators",
               issue = "27754")]
    #[inline]
    pub fn slice_shift_char(&self) -> Option<(char, &str)> {
        core_str::StrExt::slice_shift_char(self)
    }

    /// Divide one string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get mutable string slices instead, see the [`split_at_mut()`]
    /// method.
    ///
    /// [`split_at_mut()`]: #method.split_at_mut
    ///
    /// # Panics
    ///
    /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is
    /// beyond the last code point of the string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Per Martin-Löf";
    ///
    /// let (first, last) = s.split_at(3);
    ///
    /// assert_eq!("Per", first);
    /// assert_eq!(" Martin-Löf", last);
    /// ```
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at(&self, mid: usize) -> (&str, &str) {
        core_str::StrExt::split_at(self, mid)
    }

    /// Divide one mutable string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get immutable string slices instead, see the [`split_at()`] method.
    ///
    /// [`split_at()`]: #method.split_at
    ///
    /// # Panics
    ///
    /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is
    /// beyond the last code point of the string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Per Martin-Löf";
    ///
    /// let (first, last) = s.split_at(3);
    ///
    /// assert_eq!("Per", first);
    /// assert_eq!(" Martin-Löf", last);
    /// ```
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
        core_str::StrExt::split_at_mut(self, mid)
    }

    /// Returns an iterator over the `char`s of a string slice.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns such an iterator.
    ///
    /// It's important to remember that [`char`] represents a Unicode Scalar
    /// Value, and may not match your idea of what a 'character' is. Iteration
    /// over grapheme clusters may be what you actually want.
    ///
    /// [`char`]: ../primitive.char.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let word = "goodbye";
    ///
    /// let count = word.chars().count();
    /// assert_eq!(7, count);
    ///
    /// let mut chars = word.chars();
    ///
    /// assert_eq!(Some('g'), chars.next());
    /// assert_eq!(Some('o'), chars.next());
    /// assert_eq!(Some('o'), chars.next());
    /// assert_eq!(Some('d'), chars.next());
    /// assert_eq!(Some('b'), chars.next());
    /// assert_eq!(Some('y'), chars.next());
    /// assert_eq!(Some('e'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    ///
    /// Remember, `char`s may not match your human intuition about characters:
    ///
    /// ```
    /// let y = "y̆";
    ///
    /// let mut chars = y.chars();
    ///
    /// assert_eq!(Some('y'), chars.next()); // not 'y̆'
    /// assert_eq!(Some('\u{0306}'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chars(&self) -> Chars {
        core_str::StrExt::chars(self)
    }
    /// Returns an iterator over the `char`s of a string slice, and their
    /// positions.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by `char`. This method returns an iterator of both
    /// these `char`s, as well as their byte positions.
    ///
    /// The iterator yields tuples. The position is first, the `char` is
    /// second.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let word = "goodbye";
    ///
    /// let count = word.char_indices().count();
    /// assert_eq!(7, count);
    ///
    /// let mut char_indices = word.char_indices();
    ///
    /// assert_eq!(Some((0, 'g')), char_indices.next());
    /// assert_eq!(Some((1, 'o')), char_indices.next());
    /// assert_eq!(Some((2, 'o')), char_indices.next());
    /// assert_eq!(Some((3, 'd')), char_indices.next());
    /// assert_eq!(Some((4, 'b')), char_indices.next());
    /// assert_eq!(Some((5, 'y')), char_indices.next());
    /// assert_eq!(Some((6, 'e')), char_indices.next());
    ///
    /// assert_eq!(None, char_indices.next());
    /// ```
    ///
    /// Remember, `char`s may not match your human intuition about characters:
    ///
    /// ```
    /// let y = "y̆";
    ///
    /// let mut char_indices = y.char_indices();
    ///
    /// assert_eq!(Some((0, 'y')), char_indices.next()); // not (0, 'y̆')
    /// assert_eq!(Some((1, '\u{0306}')), char_indices.next());
    ///
    /// assert_eq!(None, char_indices.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn char_indices(&self) -> CharIndices {
        core_str::StrExt::char_indices(self)
    }

    /// An iterator over the bytes of a string slice.
    ///
    /// As a string slice consists of a sequence of bytes, we can iterate
    /// through a string slice by byte. This method returns such an iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut bytes = "bors".bytes();
    ///
    /// assert_eq!(Some(b'b'), bytes.next());
    /// assert_eq!(Some(b'o'), bytes.next());
    /// assert_eq!(Some(b'r'), bytes.next());
    /// assert_eq!(Some(b's'), bytes.next());
    ///
    /// assert_eq!(None, bytes.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn bytes(&self) -> Bytes {
        core_str::StrExt::bytes(self)
    }

    /// Split a string slice by whitespace.
    ///
    /// The iterator returned will return string slices that are sub-slices of
    /// the original string slice, separated by any amount of whitespace.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut iter = "A few words".split_whitespace();
    ///
    /// assert_eq!(Some("A"), iter.next());
    /// assert_eq!(Some("few"), iter.next());
    /// assert_eq!(Some("words"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// All kinds of whitespace are considered:
    ///
    /// ```
    /// let mut iter = " Mary   had\ta\u{2009}little  \n\t lamb".split_whitespace();
    /// assert_eq!(Some("Mary"), iter.next());
    /// assert_eq!(Some("had"), iter.next());
    /// assert_eq!(Some("a"), iter.next());
    /// assert_eq!(Some("little"), iter.next());
    /// assert_eq!(Some("lamb"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    #[stable(feature = "split_whitespace", since = "1.1.0")]
    #[inline]
    pub fn split_whitespace(&self) -> SplitWhitespace {
        UnicodeStr::split_whitespace(self)
    }

    /// An iterator over the lines of a string, as string slices.
    ///
    /// Lines are ended with either a newline (`\n`) or a carriage return with
    /// a line feed (`\r\n`).
    ///
    /// The final line ending is optional.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let text = "foo\r\nbar\n\nbaz\n";
    /// let mut lines = text.lines();
    ///
    /// assert_eq!(Some("foo"), lines.next());
    /// assert_eq!(Some("bar"), lines.next());
    /// assert_eq!(Some(""), lines.next());
    /// assert_eq!(Some("baz"), lines.next());
    ///
    /// assert_eq!(None, lines.next());
    /// ```
    ///
    /// The final line ending isn't required:
    ///
    /// ```
    /// let text = "foo\nbar\n\r\nbaz";
    /// let mut lines = text.lines();
    ///
    /// assert_eq!(Some("foo"), lines.next());
    /// assert_eq!(Some("bar"), lines.next());
    /// assert_eq!(Some(""), lines.next());
    /// assert_eq!(Some("baz"), lines.next());
    ///
    /// assert_eq!(None, lines.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn lines(&self) -> Lines {
        core_str::StrExt::lines(self)
    }

    /// An iterator over the lines of a string.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.4.0", reason = "use lines() instead now")]
    #[inline]
    #[allow(deprecated)]
    pub fn lines_any(&self) -> LinesAny {
        core_str::StrExt::lines_any(self)
    }

    /// Returns an iterator of `u16` over the string encoded as UTF-16.
    #[unstable(feature = "str_utf16",
               reason = "this functionality may only be provided by libunicode",
               issue = "27714")]
    pub fn utf16_units(&self) -> Utf16Units {
        Utf16Units { encoder: Utf16Encoder::new(self[..].chars()) }
    }

    /// Returns `true` if the given `&str` is a sub-slice of this string slice.
    ///
    /// Returns `false` if it's not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.contains("nana"));
    /// assert!(!bananas.contains("apples"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        core_str::StrExt::contains(self, pat)
    }

    /// Returns `true` if the given `&str` is a prefix of this string slice.
    ///
    /// Returns `false` if it's not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.starts_with("bana"));
    /// assert!(!bananas.starts_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        core_str::StrExt::starts_with(self, pat)
    }

    /// Returns `true` if the given `&str` is a suffix of this string slice.
    ///
    /// Returns `false` if not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```rust
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.ends_with("anas"));
    /// assert!(!bananas.ends_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ends_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::ends_with(self, pat)
    }

    /// Returns the byte index of the first character of this string slice that
    /// matches the pattern.
    ///
    /// Returns `None` if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find('L'), Some(0));
    /// assert_eq!(s.find('é'), Some(14));
    /// assert_eq!(s.find("Léopard"), Some(13));
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find(char::is_whitespace), Some(5));
    /// assert_eq!(s.find(char::is_lowercase), Some(1));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.find(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn find<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize> {
        core_str::StrExt::find(self, pat)
    }

    /// Returns the byte index of the last character of this string slice that
    /// matches the pattern.
    ///
    /// Returns `None` if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind('L'), Some(13));
    /// assert_eq!(s.rfind('é'), Some(14));
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind(char::is_whitespace), Some(12));
    /// assert_eq!(s.rfind(char::is_lowercase), Some(20));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.rfind(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rfind<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rfind(self, pat)
    }

    /// An iterator over substrings of this string slice, separated by
    /// characters matched by a pattern.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rsplit()`] method can be used.
    ///
    /// [`char`]: primitive.char.html
    /// [`rsplit()`]: #method.rsplit
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".split(' ').collect();
    /// assert_eq!(v, ["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: Vec<&str> = "".split('X').collect();
    /// assert_eq!(v, [""]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').collect();
    /// assert_eq!(v, ["lion", "", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".split("::").collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".split(char::is_numeric).collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    ///
    /// let v: Vec<&str> = "lionXtigerXleopard".split(char::is_uppercase).collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".split(|c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    /// ```
    ///
    /// If a string contains multiple contiguous separators, you will end up
    /// with empty strings in the output:
    ///
    /// ```
    /// let x = "||||a||b|c".to_string();
    /// let d: Vec<_> = x.split('|').collect();
    ///
    /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    /// ```
    ///
    /// This can lead to possibly surprising behavior when whitespace is used
    /// as the separator. This code is correct:
    ///
    /// ```
    /// let x = "    a  b c".to_string();
    /// let d: Vec<_> = x.split(' ').collect();
    ///
    /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    /// ```
    ///
    /// It does _not_ give you:
    ///
    /// ```rust,ignore
    /// assert_eq!(d, &["a", "b", "c"]);
    /// ```
    ///
    /// Use [`split_whitespace()`] for this behavior.
    ///
    /// [`split_whitespace()`]: #method.split_whitespace
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P> {
        core_str::StrExt::split(self, pat)
    }

    /// An iterator over substrings of the given string slice, separated by
    /// characters matched by a pattern and yielded in reverse order.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`split()`] method can be used.
    ///
    /// [`split()`]: #method.split
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".rsplit(' ').collect();
    /// assert_eq!(v, ["lamb", "little", "a", "had", "Mary"]);
    ///
    /// let v: Vec<&str> = "".rsplit('X').collect();
    /// assert_eq!(v, [""]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplit('X').collect();
    /// assert_eq!(v, ["leopard", "tiger", "", "lion"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".rsplit("::").collect();
    /// assert_eq!(v, ["leopard", "tiger", "lion"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".rsplit(|c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["ghi", "def", "abc"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rsplit<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplit<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplit(self, pat)
    }

    /// An iterator over substrings of the given string slice, separated by
    /// characters matched by a pattern.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// Equivalent to [`split()`], except that the trailing substring
    /// is skipped if empty.
    ///
    /// [`split()`]: #method.split
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    /// [`char`]: primitive.char.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rsplit_terminator()`] method can be used.
    ///
    /// [`rsplit_terminator()`]: #method.rsplit_terminator
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".split_terminator('.').collect();
    /// assert_eq!(v, ["A", "B"]);
    ///
    /// let v: Vec<&str> = "A..B..".split_terminator(".").collect();
    /// assert_eq!(v, ["A", "", "B", ""]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P> {
        core_str::StrExt::split_terminator(self, pat)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern and yielded in reverse order.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// Equivalent to `split`, except that the trailing substring is
    /// skipped if empty.
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a
    /// reverse search, and it will be double ended if a forward/reverse
    /// search yields the same elements.
    ///
    /// For iterating from the front, the [`split_terminator()`] method can be
    /// used.
    ///
    /// [`split_terminator()`]: #method.split_terminator
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".rsplit_terminator('.').collect();
    /// assert_eq!(v, ["B", "A"]);
    ///
    /// let v: Vec<&str> = "A..B..".rsplit_terminator(".").collect();
    /// assert_eq!(v, ["", "B", "", "A"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rsplit_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplitTerminator<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplit_terminator(self, pat)
    }

    /// An iterator over substrings of the given string slice, separated by a
    /// pattern, restricted to returning at most `count` items.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// string slice.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is
    /// not efficient to support.
    ///
    /// If the pattern allows a reverse search, the [`rsplitn()`] method can be
    /// used.
    ///
    /// [`rsplitn()`]: #method.rsplitn
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lambda".splitn(3, ' ').collect();
    /// assert_eq!(v, ["Mary", "had", "a little lambda"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".splitn(3, "X").collect();
    /// assert_eq!(v, ["lion", "", "tigerXleopard"]);
    ///
    /// let v: Vec<&str> = "abcXdef".splitn(1, 'X').collect();
    /// assert_eq!(v, ["abcXdef"]);
    ///
    /// let v: Vec<&str> = "".splitn(1, 'X').collect();
    /// assert_eq!(v, [""]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".splitn(2, |c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["abc", "defXghi"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn splitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> SplitN<'a, P> {
        core_str::StrExt::splitn(self, count, pat)
    }

    /// An iterator over substrings of this string slice, separated by a
    /// pattern, starting from the end of the string, restricted to returning
    /// at most `count` items.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// string slice.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines the split.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is not
    /// efficient to support.
    ///
    /// For splitting from the front, the [`splitn()`] method can be used.
    ///
    /// [`splitn()`]: #method.splitn
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".rsplitn(3, ' ').collect();
    /// assert_eq!(v, ["lamb", "little", "Mary had a"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplitn(3, 'X').collect();
    /// assert_eq!(v, ["leopard", "tiger", "lionX"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".rsplitn(2, "::").collect();
    /// assert_eq!(v, ["leopard", "lion::tiger"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".rsplitn(2, |c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["ghi", "abc1def"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rsplitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> RSplitN<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplitn(self, count, pat)
    }

    /// An iterator over the matches of a pattern within the given string
    /// slice.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines if a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    /// [`char`]: primitive.char.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatches()`] method can be used.
    ///
    /// [`rmatches()`]: #method.rmatches
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".matches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".matches(char::is_numeric).collect();
    /// assert_eq!(v, ["1", "2", "3"]);
    /// ```
    #[stable(feature = "str_matches", since = "1.2.0")]
    pub fn matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> Matches<'a, P> {
        core_str::StrExt::matches(self, pat)
    }

    /// An iterator over the matches of a pattern within this string slice,
    /// yielded in reverse order.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`matches()`] method can be used.
    ///
    /// [`matches`]: #method.matches
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".rmatches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".rmatches(char::is_numeric).collect();
    /// assert_eq!(v, ["3", "2", "1"]);
    /// ```
    #[stable(feature = "str_matches", since = "1.2.0")]
    pub fn rmatches<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatches<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rmatches(self, pat)
    }

    /// An iterator over the disjoint matches of a pattern within this string
    /// slice as well as the index that the match starts at.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the first match are returned.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines
    /// if a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatch_indices()`] method can be used.
    ///
    /// [`rmatch_indices()`]: #method.rmatch_indices
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<_> = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, [(0, "abc"), (6, "abc"), (12, "abc")]);
    ///
    /// let v: Vec<_> = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, [(1, "abc"), (4, "abc")]);
    ///
    /// let v: Vec<_> = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, [(0, "aba")]); // only the first `aba`
    /// ```
    #[stable(feature = "str_match_indices", since = "1.5.0")]
    pub fn match_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> MatchIndices<'a, P> {
        core_str::StrExt::match_indices(self, pat)
    }

    /// An iterator over the disjoint matches of a pattern within `self`,
    /// yielded in reverse order along with the index of the match.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the last match are returned.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if a
    /// character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a `[DoubleEndedIterator]` if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`match_indices()`] method can be used.
    ///
    /// [`match_indices()`]: #method.match_indices
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<_> = "abcXXXabcYYYabc".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(12, "abc"), (6, "abc"), (0, "abc")]);
    ///
    /// let v: Vec<_> = "1abcabc2".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(4, "abc"), (1, "abc")]);
    ///
    /// let v: Vec<_> = "ababa".rmatch_indices("aba").collect();
    /// assert_eq!(v, [(2, "aba")]); // only the last `aba`
    /// ```
    #[stable(feature = "str_match_indices", since = "1.5.0")]
    pub fn rmatch_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatchIndices<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rmatch_indices(self, pat)
    }

    /// Returns a string slice with leading and trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!("Hello\tworld", s.trim());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim(&self) -> &str {
        UnicodeStr::trim(self)
    }

    /// Returns a string slice with leading whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!("Hello\tworld\t", s.trim_left());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_left(&self) -> &str {
        UnicodeStr::trim_left(self)
    }

    /// Returns a string slice with trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!(" Hello\tworld", s.trim_right());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_right(&self) -> &str {
        UnicodeStr::trim_right(self)
    }

    /// Returns a string slice with all prefixes and suffixes that match a
    /// pattern repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines
    /// if a character matches.
    ///
    /// [`char`]: primtive.char.html
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    /// assert_eq!("123foo1bar123".trim_matches(char::is_numeric), "foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_matches(x), "foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1foo1barXX".trim_matches(|c| c == '1' || c == 'X'), "foo1bar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: DoubleEndedSearcher<'a>
    {
        core_str::StrExt::trim_matches(self, pat)
    }

    /// Returns a string slice with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
    /// assert_eq!("123foo1bar123".trim_left_matches(char::is_numeric), "foo1bar123");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_left_matches(x), "foo1bar12");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_left_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str {
        core_str::StrExt::trim_left_matches(self, pat)
    }

    /// Returns a string slice with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines if a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
    /// assert_eq!("123foo1bar123".trim_right_matches(char::is_numeric), "123foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_right_matches(x), "12foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1fooX".trim_left_matches(|c| c == '1' || c == 'X'), "fooX");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_right_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::trim_right_matches(self, pat)
    }

    /// Parses this string slice into another type.
    ///
    /// Because `parse()` is so general, it can cause problems with type
    /// inference. As such, `parse()` is one of the few times you'll see
    /// the syntax affectionately known as the 'turbofish': `::<>`. This
    /// helps the inference algorithm understand specifically which type
    /// you're trying to parse into.
    ///
    /// `parse()` can parse any type that implements the [`FromStr`] trait.
    ///
    /// [`FromStr`]: trait.FromStr.html
    ///
    /// # Failure
    ///
    /// Will return `Err` if it's not possible to parse this string slice into
    /// the desired type.
    ///
    /// # Example
    ///
    /// Basic usage
    ///
    /// ```
    /// let four: u32 = "4".parse().unwrap();
    ///
    /// assert_eq!(4, four);
    /// ```
    ///
    /// Using the 'turbofish' instead of annotationg `four`:
    ///
    /// ```
    /// let four = "4".parse::<u32>();
    ///
    /// assert_eq!(Ok(4), four);
    /// ```
    ///
    /// Failing to parse:
    ///
    /// ```
    /// let nope = "j".parse::<u32>();
    ///
    /// assert!(nope.is_err());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
        core_str::StrExt::parse(self)
    }

    /// Replaces all occurrences of one string with another.
    ///
    /// `replace` creates a new [`String`], and copies the data from this string slice into it.
    /// While doing so, it attempts to find a sub-`&str`. If it finds it, it replaces it with
    /// the replacement string slice.
    ///
    /// [`String`]: string/struct.String.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "this is old";
    ///
    /// assert_eq!("this is new", s.replace("old", "new"));
    /// ```
    ///
    /// When a `&str` isn't found:
    ///
    /// ```
    /// let s = "this is old";
    /// assert_eq!(s, s.replace("cookie monster", "little lamb"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn replace(&self, from: &str, to: &str) -> String {
        let mut result = String::new();
        let mut last_end = 0;
        for (start, part) in self.match_indices(from) {
            result.push_str(unsafe { self.slice_unchecked(last_end, start) });
            result.push_str(to);
            last_end = start + part.len();
        }
        result.push_str(unsafe { self.slice_unchecked(last_end, self.len()) });
        result
    }

    /// Returns the lowercase equivalent of this string slice, as a new `String`.
    ///
    /// 'Lowercase' is defined according to the terms of the Unicode Derived Core Property
    /// `Lowercase`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "HELLO";
    ///
    /// assert_eq!("hello", s.to_lowercase());
    /// ```
    ///
    /// A tricky example, with sigma:
    ///
    /// ```
    /// let sigma = "Σ";
    ///
    /// assert_eq!("σ", sigma.to_lowercase());
    ///
    /// // but at the end of a word, it's ς, not σ:
    /// let odysseus = "ὈΔΥΣΣΕΎΣ";
    ///
    /// assert_eq!("ὀδυσσεύς", odysseus.to_lowercase());
    /// ```
    ///
    /// Languages without case are not changed:
    ///
    /// ```
    /// let new_year = "农历新年";
    ///
    /// assert_eq!(new_year, new_year.to_lowercase());
    /// ```
    #[stable(feature = "unicode_case_mapping", since = "1.2.0")]
    pub fn to_lowercase(&self) -> String {
        let mut s = String::with_capacity(self.len());
        for (i, c) in self[..].char_indices() {
            if c == 'Σ' {
                // Σ maps to σ, except at the end of a word where it maps to ς.
                // This is the only conditional (contextual) but language-independent mapping
                // in `SpecialCasing.txt`,
                // so hard-code it rather than have a generic "condition" mechanim.
                // See https://github.com/rust-lang/rust/issues/26035
                map_uppercase_sigma(self, i, &mut s)
            } else {
                s.extend(c.to_lowercase());
            }
        }
        return s;

        fn map_uppercase_sigma(from: &str, i: usize, to: &mut String) {
            // See http://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
            // for the definition of `Final_Sigma`.
            debug_assert!('Σ'.len_utf8() == 2);
            let is_word_final = case_ignoreable_then_cased(from[..i].chars().rev()) &&
                                !case_ignoreable_then_cased(from[i + 2..].chars());
            to.push_str(if is_word_final {
                "ς"
            } else {
                "σ"
            });
        }

        fn case_ignoreable_then_cased<I: Iterator<Item = char>>(iter: I) -> bool {
            use rustc_unicode::derived_property::{Cased, Case_Ignorable};
            match iter.skip_while(|&c| Case_Ignorable(c)).next() {
                Some(c) => Cased(c),
                None => false,
            }
        }
    }

    /// Returns the uppercase equivalent of this string slice, as a new `String`.
    ///
    /// 'Uppercase' is defined according to the terms of the Unicode Derived Core Property
    /// `Uppercase`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "hello";
    ///
    /// assert_eq!("HELLO", s.to_uppercase());
    /// ```
    ///
    /// Scripts without case are not changed:
    ///
    /// ```
    /// let new_year = "农历新年";
    ///
    /// assert_eq!(new_year, new_year.to_uppercase());
    /// ```
    #[stable(feature = "unicode_case_mapping", since = "1.2.0")]
    pub fn to_uppercase(&self) -> String {
        let mut s = String::with_capacity(self.len());
        s.extend(self.chars().flat_map(|c| c.to_uppercase()));
        return s;
    }

    /// Escapes each char in `s` with `char::escape_default`.
    #[unstable(feature = "str_escape",
               reason = "return type may change to be an iterator",
               issue = "27791")]
    pub fn escape_default(&self) -> String {
        self.chars().flat_map(|c| c.escape_default()).collect()
    }

    /// Escapes each char in `s` with `char::escape_unicode`.
    #[unstable(feature = "str_escape",
               reason = "return type may change to be an iterator",
               issue = "27791")]
    pub fn escape_unicode(&self) -> String {
        self.chars().flat_map(|c| c.escape_unicode()).collect()
    }

    /// Converts a `Box<str>` into a `String` without copying or allocating.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let string = String::from("birthday gift");
    /// let boxed_str = string.clone().into_boxed_str();
    ///
    /// assert_eq!(boxed_str.into_string(), string);
    /// ```
    #[stable(feature = "box_str", since = "1.4.0")]
    pub fn into_string(self: Box<str>) -> String {
        unsafe {
            let slice = mem::transmute::<Box<str>, Box<[u8]>>(self);
            String::from_utf8_unchecked(slice.into_vec())
        }
    }
}
