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

/// Any string that can be represented as a slice.
#[lang = "str"]
#[cfg(not(test))]
impl str {
    /// Returns the length of `self` in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("foo".len(), 3);
    /// assert_eq!("ƒoo".len(), 4); // fancy f!
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
    /// ```
    /// assert!("".is_empty());
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

    /// Converts `self` to a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("bors".as_bytes(), b"bors");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        core_str::StrExt::as_bytes(self)
    }

    /// Returns a raw pointer to the `&str`'s buffer.
    ///
    /// The caller must ensure that the string outlives this pointer, and
    /// that it is not
    /// reallocated (e.g. by pushing to the string).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Hello";
    /// let p = s.as_ptr();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        core_str::StrExt::as_ptr(self)
    }

    /// Takes a bytewise slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// # Safety
    ///
    /// Caller must check both UTF-8 sequence boundaries and the boundaries
    /// of the entire slice as well.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// unsafe {
    ///     assert_eq!(s.slice_unchecked(0, 21), "Löwe 老虎 Léopard");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        core_str::StrExt::slice_unchecked(self, begin, end)
    }

    /// Takes a bytewise mutable slice from a string.
    ///
    /// Same as `slice_unchecked`, but works with `&mut str` instead of `&str`.
    #[stable(feature = "str_slice_mut", since = "1.5.0")]
    #[inline]
    pub unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str {
        core_str::StrExt::slice_mut_unchecked(self, begin, end)
    }

    /// Given a byte position, return the next code point and its index.
    ///
    /// This can be used to iterate over the Unicode code points of a string.
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

    /// Given a byte position, return the previous `char` and its position.
    ///
    /// This function can be used to iterate over a Unicode code points in reverse.
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

    /// Given a byte position, return the `char` at that position.
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

    /// Given a byte position, return the `char` at that position, counting
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

    /// Retrieves the first code point from a `&str` and returns it.
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
    /// The index `mid` is a byte offset from the start of the string
    /// that must be on a `char` boundary.
    ///
    /// Return slices `&self[..mid]` and `&self[mid..]`.
    ///
    /// # Panics
    ///
    /// Panics if `mid` is beyond the last code point of the string,
    /// or if it is not on a `char` boundary.
    ///
    /// # Examples
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let first_space = s.find(' ').unwrap_or(s.len());
    /// let (a, b) = s.split_at(first_space);
    ///
    /// assert_eq!(a, "Löwe");
    /// assert_eq!(b, " 老虎 Léopard");
    /// ```
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at(&self, mid: usize) -> (&str, &str) {
        core_str::StrExt::split_at(self, mid)
    }

    /// Divide one mutable string slice into two at an index.
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
        core_str::StrExt::split_at_mut(self, mid)
    }

    /// An iterator over the code points of `self`.
    ///
    /// In Unicode relationship between code points and characters is complex.
    /// A single character may be composed of multiple code points
    /// (e.g. diacritical marks added to a letter), and a single code point
    /// (e.g. Hangul syllable) may contain multiple characters.
    ///
    /// For iteration over human-readable characters a grapheme cluster iterator
    /// may be more appropriate. See the [unicode-segmentation crate][1].
    ///
    /// [1]: https://crates.io/crates/unicode-segmentation
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<char> = "ASCII żółć 🇨🇭 한".chars().collect();
    ///
    /// assert_eq!(v, ['A', 'S', 'C', 'I', 'I', ' ',
    ///     'z', '\u{307}', 'o', '\u{301}', 'ł', 'c', '\u{301}', ' ',
    ///     '\u{1f1e8}', '\u{1f1ed}', ' ', '한']);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chars(&self) -> Chars {
        core_str::StrExt::chars(self)
    }

    /// An iterator over the `char`s of `self` and their byte offsets.
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<(usize, char)> = "A🇨🇭".char_indices().collect();
    /// let b = vec![(0, 'A'), (1, '\u{1f1e8}'), (5, '\u{1f1ed}')];
    ///
    /// assert_eq!(v, b);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn char_indices(&self) -> CharIndices {
        core_str::StrExt::char_indices(self)
    }

    /// An iterator over the bytes of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<u8> = "bors".bytes().collect();
    ///
    /// assert_eq!(v, b"bors".to_vec());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn bytes(&self) -> Bytes {
        core_str::StrExt::bytes(self)
    }

    /// An iterator over the non-empty substrings of `self` which contain no whitespace,
    /// and which are separated by any amount of whitespace.
    ///
    /// # Examples
    ///
    /// ```
    /// let some_words = " Mary   had\ta\u{2009}little  \n\t lamb";
    /// let v: Vec<&str> = some_words.split_whitespace().collect();
    ///
    /// assert_eq!(v, ["Mary", "had", "a", "little", "lamb"]);
    /// ```
    #[stable(feature = "split_whitespace", since = "1.1.0")]
    #[inline]
    pub fn split_whitespace(&self) -> SplitWhitespace {
        UnicodeStr::split_whitespace(self)
    }

    /// An iterator over the lines of a string, separated by `\n` or `\r\n`.
    ///
    /// This does not include the empty string after a trailing newline or CRLF.
    ///
    /// # Examples
    ///
    /// ```
    /// let four_lines = "foo\nbar\n\r\nbaz";
    /// let v: Vec<&str> = four_lines.lines().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    ///
    /// Leaving off the trailing character:
    ///
    /// ```
    /// let four_lines = "foo\r\nbar\n\nbaz\n";
    /// let v: Vec<&str> = four_lines.lines().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn lines(&self) -> Lines {
        core_str::StrExt::lines(self)
    }

    /// An iterator over the lines of a string, separated by either
    /// `\n` or `\r\n`.
    ///
    /// As with `.lines()`, this does not include an empty trailing line.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(deprecated)]
    /// let four_lines = "foo\r\nbar\n\r\nbaz";
    /// let v: Vec<&str> = four_lines.lines_any().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    ///
    /// Leaving off the trailing character:
    ///
    /// ```
    /// # #![allow(deprecated)]
    /// let four_lines = "foo\r\nbar\n\r\nbaz\n";
    /// let v: Vec<&str> = four_lines.lines_any().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
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

    /// Returns `true` if `self` contains another `&str`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!("bananas".contains("nana"));
    ///
    /// assert!(!"bananas".contains("foobar"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        core_str::StrExt::contains(self, pat)
    }

    /// Returns `true` if the given `&str` is a prefix of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!("banana".starts_with("ba"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        core_str::StrExt::starts_with(self, pat)
    }

    /// Returns true if the given `&str` is a suffix of the string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert!("banana".ends_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ends_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::ends_with(self, pat)
    }

    /// Returns the byte index of the first character of `self` that matches
    /// the pattern, if it
    /// exists.
    ///
    /// Returns `None` if it doesn't exist.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines if a character matches.
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
    ///
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

    /// Returns the byte index of the last character of `self` that
    /// matches the pattern, if it
    /// exists.
    ///
    /// Returns `None` if it doesn't exist.
    ///
    /// The pattern can be a simple `&str`, `char`,
    /// or a closure that determines if a character matches.
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

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split. Additional libraries might provide more complex
    /// patterns like regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be double ended if the pattern allows a
    /// reverse search and forward/reverse search yields the same elements.
    /// This is true for, eg, `char` but not
    /// for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, `rsplit()` can be used.
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
    /// Use [`.split_whitespace()`][split_whitespace] for this behavior.
    ///
    /// [split_whitespace]: #method.split_whitespace
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P> {
        core_str::StrExt::split(self, pat)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern and yielded in reverse order.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a
    /// reverse search,
    /// and it will be double ended if a forward/reverse search yields
    /// the same elements.
    ///
    /// For iterating from the front, `split()` can be used.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```rust
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

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns
    /// like regular expressions.
    ///
    /// Equivalent to `split`, except that the trailing substring
    /// is skipped if empty.
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be double ended if the pattern allows a
    /// reverse search
    /// and forward/reverse search yields the same elements. This is true
    /// for, eg, `char` but not for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, `rsplit_terminator()` can be used.
    ///
    /// # Examples
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
    /// For iterating from the front, `split_terminator()` can be used.
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

    /// An iterator over substrings of `self`, separated by a pattern,
    /// restricted to returning
    /// at most `count` items.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// string.
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is
    /// not efficient to support.
    ///
    /// If the pattern allows a reverse search, `rsplitn()` can be used.
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

    /// An iterator over substrings of `self`, separated by a pattern,
    /// starting from the end of the string, restricted to returning
    /// at most `count` items.
    ///
    /// The last element returned, if any, will contain the remainder of the
    /// string.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is not
    /// efficient to support.
    ///
    /// `splitn()` can be used for splitting from the front.
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

    /// An iterator over the matches of a pattern within `self`.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines if a character matches.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be double ended if the pattern allows
    /// a reverse search
    /// and forward/reverse search yields the same elements. This is true
    /// for, eg, `char` but not
    /// for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, `rmatches()` can be used.
    ///
    /// # Examples
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

    /// An iterator over the matches of a pattern within `self`, yielded in
    /// reverse order.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines if a character matches.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a
    /// reverse search,
    /// and it will be double ended if a forward/reverse search yields
    /// the same elements.
    ///
    /// For iterating from the front, `matches()` can be used.
    ///
    /// # Examples
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

    /// An iterator over the disjoint matches of a pattern within `self` as well
    /// as the index that the match starts at.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the first match are returned.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that determines
    /// if a character matches. Additional libraries might provide more complex
    /// patterns like regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be double ended if the pattern allows a
    /// reverse search and forward/reverse search yields the same elements. This
    /// is true for, eg, `char` but not for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, `rmatch_indices()` can be used.
    ///
    /// # Examples
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
    /// The pattern can be a simple `&str`, `char`, or a closure that determines
    /// if a character matches. Additional libraries might provide more complex
    /// patterns like regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be double ended if a forward/reverse search yields
    /// the same elements.
    ///
    /// For iterating from the front, `match_indices()` can be used.
    ///
    /// # Examples
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

    /// Returns a `&str` with leading and trailing whitespace removed.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    /// assert_eq!(s.trim(), "Hello\tworld");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim(&self) -> &str {
        UnicodeStr::trim(self)
    }

    /// Returns a `&str` with leading whitespace removed.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    /// assert_eq!(s.trim_left(), "Hello\tworld\t");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_left(&self) -> &str {
        UnicodeStr::trim_left(self)
    }

    /// Returns a `&str` with trailing whitespace removed.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    /// assert_eq!(s.trim_right(), " Hello\tworld");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_right(&self) -> &str {
        UnicodeStr::trim_right(self)
    }

    /// Returns a string with all pre- and suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a simple `char`, or a closure that determines
    /// if a character matches.
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

    /// Returns a string with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines if a character matches.
    ///
    /// # Examples
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

    /// Returns a string with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines if a character matches.
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

    /// Parses `self` into the specified type.
    ///
    /// # Failure
    ///
    /// Will return `Err` if it's not possible to parse `self` into the type.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!("4".parse::<u32>(), Ok(4));
    /// ```
    ///
    /// Failing:
    ///
    /// ```
    /// assert!("j".parse::<u32>().is_err());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
        core_str::StrExt::parse(self)
    }

    /// Replaces all occurrences of one string with another.
    ///
    /// `replace` takes two arguments, a sub-`&str` to find in `self`, and a
    /// second `&str` to
    /// replace it with. If the original `&str` isn't found, no change occurs.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "this is old";
    ///
    /// assert_eq!(s.replace("old", "new"), "this is new");
    /// ```
    ///
    /// When a `&str` isn't found:
    ///
    /// ```
    /// let s = "this is old";
    /// assert_eq!(s.replace("cookie monster", "little lamb"), s);
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

    /// Returns the lowercase equivalent of this string.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "HELLO";
    /// assert_eq!(s.to_lowercase(), "hello");
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

    /// Returns the uppercase equivalent of this string.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "hello";
    /// assert_eq!(s.to_uppercase(), "HELLO");
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

    /// Converts the `Box<str>` into a `String` without copying or allocating.
    #[stable(feature = "box_str", since = "1.4.0")]
    pub fn into_string(self: Box<str>) -> String {
        unsafe {
            let slice = mem::transmute::<Box<str>, Box<[u8]>>(self);
            String::from_utf8_unchecked(slice.into_vec())
        }
    }
}
