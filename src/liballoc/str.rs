// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unicode string slices.
//!
//! The `&str` type is one of the two main string types, the other being `String`.
//! Unlike its `String` counterpart, its contents are borrowed.
//!
//! # Basic Usage
//!
//! A basic string declaration of `&str` type:
//!
//! ```
//! let hello_world = "Hello, World!";
//! ```
//!
//! Here we have declared a string literal, also known as a string slice.
//! String literals have a static lifetime, which means the string `hello_world`
//! is guaranteed to be valid for the duration of the entire program.
//! We can explicitly specify `hello_world`'s lifetime as well:
//!
//! ```
//! let hello_world: &'static str = "Hello, world!";
//! ```
//!
//! *[See also the `str` primitive type](../../std/primitive.str.html).*

#![stable(feature = "rust1", since = "1.0.0")]

// Many of the usings in this module are only used in the test configuration.
// It's cleaner to just turn off the unused_imports warning than to fix them.
#![allow(unused_imports)]

use core::fmt;
use core::str as core_str;
use core::str::pattern::Pattern;
use core::str::pattern::{Searcher, ReverseSearcher, DoubleEndedSearcher};
use core::mem;
use core::iter::FusedIterator;
use std_unicode::str::{UnicodeStr, Utf16Encoder};

use vec_deque::VecDeque;
use borrow::{Borrow, ToOwned};
use string::String;
use std_unicode;
use vec::Vec;
use slice::{SliceConcatExt, SliceIndex};
use boxed::Box;

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{FromStr, Utf8Error};
#[allow(deprecated)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Lines, LinesAny};
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
pub use core::str::{from_utf8, from_utf8_mut, Chars, CharIndices, Bytes};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{from_utf8_unchecked, from_utf8_unchecked_mut, ParseBoolError};
#[stable(feature = "rust1", since = "1.0.0")]
pub use std_unicode::str::SplitWhitespace;
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

/// An iterator of [`u16`] over the string encoded as UTF-16.
///
/// [`u16`]: ../../std/primitive.u16.html
///
/// This struct is created by the [`encode_utf16`] method on [`str`].
/// See its documentation for more.
///
/// [`encode_utf16`]: ../../std/primitive.str.html#method.encode_utf16
/// [`str`]: ../../std/primitive.str.html
#[derive(Clone)]
#[stable(feature = "encode_utf16", since = "1.8.0")]
pub struct EncodeUtf16<'a> {
    encoder: Utf16Encoder<Chars<'a>>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<'a> fmt::Debug for EncodeUtf16<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("EncodeUtf16 { .. }")
    }
}

#[stable(feature = "encode_utf16", since = "1.8.0")]
impl<'a> Iterator for EncodeUtf16<'a> {
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

#[unstable(feature = "fused", issue = "35602")]
impl<'a> FusedIterator for EncodeUtf16<'a> {}

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

    fn clone_into(&self, target: &mut String) {
        let mut b = mem::replace(target, String::new()).into_bytes();
        self.as_bytes().clone_into(&mut b);
        *target = unsafe { String::from_utf8_unchecked(b) }
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

    /// Returns `true` if `self` has a length of zero bytes.
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
    #[stable(feature = "is_char_boundary", since = "1.9.0")]
    #[inline]
    pub fn is_char_boundary(&self, index: usize) -> bool {
        core_str::StrExt::is_char_boundary(self, index)
    }

    /// Converts a string slice to a byte slice. To convert the byte slice back
    /// into a string slice, use the [`str::from_utf8`] function.
    ///
    /// [`str::from_utf8`]: ./str/fn.from_utf8.html
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

    /// Converts a mutable string slice to a mutable byte slice. To convert the
    /// mutable byte slice back into a mutable string slice, use the
    /// [`str::from_utf8_mut`] function.
    ///
    /// [`str::from_utf8_mut`]: ./str/fn.from_utf8_mut.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = String::from("Hello");
    /// let bytes = unsafe { s.as_bytes_mut() };
    ///
    /// assert_eq!(b"Hello", bytes);
    /// ```
    ///
    /// Mutability:
    ///
    /// ```
    /// let mut s = String::from("🗻∈🌏");
    ///
    /// unsafe {
    ///     let bytes = s.as_bytes_mut();
    ///
    ///     bytes[0] = 0xF0;
    ///     bytes[1] = 0x9F;
    ///     bytes[2] = 0x8D;
    ///     bytes[3] = 0x94;
    /// }
    ///
    /// assert_eq!("🍔∈🌏", s);
    /// ```
    #[stable(feature = "str_mut_extras", since = "1.20.0")]
    #[inline(always)]
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        core_str::StrExt::as_bytes_mut(self)
    }

    /// Converts a string slice to a raw pointer.
    ///
    /// As string slices are a slice of bytes, the raw pointer points to a
    /// [`u8`]. This pointer will be pointing to the first byte of the string
    /// slice.
    ///
    /// [`u8`]: primitive.u8.html
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

    /// Returns a subslice of `str`.
    ///
    /// This is the non-panicking alternative to indexing the `str`. Returns
    /// [`None`] whenever equivalent indexing operation would panic.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let v = String::from("🗻∈🌏");
    ///
    /// assert_eq!(Some("🗻"), v.get(0..4));
    ///
    /// // indices not on UTF-8 sequence boundaries
    /// assert!(v.get(1..).is_none());
    /// assert!(v.get(..8).is_none());
    ///
    /// // out of bounds
    /// assert!(v.get(..42).is_none());
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub fn get<I: SliceIndex<str>>(&self, i: I) -> Option<&I::Output> {
        core_str::StrExt::get(self, i)
    }

    /// Returns a mutable subslice of `str`.
    ///
    /// This is the non-panicking alternative to indexing the `str`. Returns
    /// [`None`] whenever equivalent indexing operation would panic.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::from("hello");
    /// // correct length
    /// assert!(v.get_mut(0..5).is_some());
    /// // out of bounds
    /// assert!(v.get_mut(..42).is_none());
    /// assert_eq!(Some("he"), v.get_mut(0..2).map(|v| &*v));
    ///
    /// assert_eq!("hello", v);
    /// {
    ///     let s = v.get_mut(0..2);
    ///     let s = s.map(|s| {
    ///         s.make_ascii_uppercase();
    ///         &*s
    ///     });
    ///     assert_eq!(Some("HE"), s);
    /// }
    /// assert_eq!("HEllo", v);
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub fn get_mut<I: SliceIndex<str>>(&mut self, i: I) -> Option<&mut I::Output> {
        core_str::StrExt::get_mut(self, i)
    }

    /// Returns a unchecked subslice of `str`.
    ///
    /// This is the unchecked alternative to indexing the `str`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The starting index must come before the ending index;
    /// * Indexes must be within bounds of the original slice;
    /// * Indexes must lie on UTF-8 sequence boundaries.
    ///
    /// Failing that, the returned string slice may reference invalid memory or
    /// violate the invariants communicated by the `str` type.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = "🗻∈🌏";
    /// unsafe {
    ///     assert_eq!("🗻", v.get_unchecked(0..4));
    ///     assert_eq!("∈", v.get_unchecked(4..7));
    ///     assert_eq!("🌏", v.get_unchecked(7..11));
    /// }
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<str>>(&self, i: I) -> &I::Output {
        core_str::StrExt::get_unchecked(self, i)
    }

    /// Returns a mutable, unchecked subslice of `str`.
    ///
    /// This is the unchecked alternative to indexing the `str`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The starting index must come before the ending index;
    /// * Indexes must be within bounds of the original slice;
    /// * Indexes must lie on UTF-8 sequence boundaries.
    ///
    /// Failing that, the returned string slice may reference invalid memory or
    /// violate the invariants communicated by the `str` type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::from("🗻∈🌏");
    /// unsafe {
    ///     assert_eq!("🗻", v.get_unchecked_mut(0..4));
    ///     assert_eq!("∈", v.get_unchecked_mut(4..7));
    ///     assert_eq!("🌏", v.get_unchecked_mut(7..11));
    /// }
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub unsafe fn get_unchecked_mut<I: SliceIndex<str>>(&mut self, i: I) -> &mut I::Output {
        core_str::StrExt::get_unchecked_mut(self, i)
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`str`] and [`Index`].
    ///
    /// [`str`]: primitive.str.html
    /// [`Index`]: ops/trait.Index.html
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get a mutable string slice instead, see the
    /// [`slice_mut_unchecked`] method.
    ///
    /// [`slice_mut_unchecked`]: #method.slice_mut_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisfied:
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
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`str`] and [`IndexMut`].
    ///
    /// [`str`]: primitive.str.html
    /// [`IndexMut`]: ops/trait.IndexMut.html
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get an immutable string slice instead, see the
    /// [`slice_unchecked`] method.
    ///
    /// [`slice_unchecked`]: #method.slice_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisfied:
    ///
    /// * `begin` must come before `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    #[stable(feature = "str_slice_mut", since = "1.5.0")]
    #[inline]
    pub unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str {
        core_str::StrExt::slice_mut_unchecked(self, begin, end)
    }

    /// Divide one string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get mutable string slices instead, see the [`split_at_mut`]
    /// method.
    ///
    /// [`split_at_mut`]: #method.split_at_mut
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
    /// To get immutable string slices instead, see the [`split_at`] method.
    ///
    /// [`split_at`]: #method.split_at
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
    /// let mut s = "Per Martin-Löf".to_string();
    /// {
    ///     let (first, last) = s.split_at_mut(3);
    ///     first.make_ascii_uppercase();
    ///     assert_eq!("PER", first);
    ///     assert_eq!(" Martin-Löf", last);
    /// }
    /// assert_eq!("PER Martin-Löf", s);
    /// ```
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
        core_str::StrExt::split_at_mut(self, mid)
    }

    /// Returns an iterator over the [`char`]s of a string slice.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns such an iterator.
    ///
    /// It's important to remember that [`char`] represents a Unicode Scalar
    /// Value, and may not match your idea of what a 'character' is. Iteration
    /// over grapheme clusters may be what you actually want.
    ///
    /// [`char`]: primitive.char.html
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
    /// Remember, [`char`]s may not match your human intuition about characters:
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
    /// Returns an iterator over the [`char`]s of a string slice, and their
    /// positions.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns an iterator of both
    /// these [`char`]s, as well as their byte positions.
    ///
    /// The iterator yields tuples. The position is first, the [`char`] is
    /// second.
    ///
    /// [`char`]: primitive.char.html
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
    /// Remember, [`char`]s may not match your human intuition about characters:
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
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let text = "Zażółć gęślą jaźń";
    ///
    /// let utf8_len = text.len();
    /// let utf16_len = text.encode_utf16().count();
    ///
    /// assert!(utf16_len <= utf8_len);
    /// ```
    #[stable(feature = "encode_utf16", since = "1.8.0")]
    pub fn encode_utf16(&self) -> EncodeUtf16 {
        EncodeUtf16 { encoder: Utf16Encoder::new(self[..].chars()) }
    }

    /// Returns `true` if the given pattern matches a sub-slice of
    /// this string slice.
    ///
    /// Returns `false` if it does not.
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
    #[inline]
    pub fn contains<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        core_str::StrExt::contains(self, pat)
    }

    /// Returns `true` if the given pattern matches a prefix of this
    /// string slice.
    ///
    /// Returns `false` if it does not.
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

    /// Returns `true` if the given pattern matches a suffix of this
    /// string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
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
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    /// [`None`]: option/enum.Option.html#variant.None
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
    /// More complex patterns using point-free style and closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find(char::is_whitespace), Some(5));
    /// assert_eq!(s.find(char::is_lowercase), Some(1));
    /// assert_eq!(s.find(|c: char| c.is_whitespace() || c.is_lowercase()), Some(1));
    /// assert_eq!(s.find(|c: char| (c < 'o') && (c > 'a')), Some(4));
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
    #[inline]
    pub fn find<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize> {
        core_str::StrExt::find(self, pat)
    }

    /// Returns the byte index of the last character of this string slice that
    /// matches the pattern.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    /// [`None`]: option/enum.Option.html#variant.None
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
    #[inline]
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
    /// from a forward search, the [`rsplit`] method can be used.
    ///
    /// [`char`]: primitive.char.html
    /// [`rsplit`]: #method.rsplit
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
    /// Contiguous separators are separated by the empty string.
    ///
    /// ```
    /// let x = "(///)".to_string();
    /// let d: Vec<_> = x.split('/').collect();
    ///
    /// assert_eq!(d, &["(", "", "", ")"]);
    /// ```
    ///
    /// Separators at the start or end of a string are neighbored
    /// by empty strings.
    ///
    /// ```
    /// let d: Vec<_> = "010".split("0").collect();
    /// assert_eq!(d, &["", "1", ""]);
    /// ```
    ///
    /// When the empty string is used as a separator, it separates
    /// every character in the string, along with the beginning
    /// and end of the string.
    ///
    /// ```
    /// let f: Vec<_> = "rust".split("").collect();
    /// assert_eq!(f, &["", "r", "u", "s", "t", ""]);
    /// ```
    ///
    /// Contiguous separators can lead to possibly surprising behavior
    /// when whitespace is used as the separator. This code is correct:
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
    /// ```,ignore
    /// assert_eq!(d, &["a", "b", "c"]);
    /// ```
    ///
    /// Use [`split_whitespace`] for this behavior.
    ///
    /// [`split_whitespace`]: #method.split_whitespace
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
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
    /// For iterating from the front, the [`split`] method can be used.
    ///
    /// [`split`]: #method.split
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
    #[inline]
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
    /// Equivalent to [`split`], except that the trailing substring
    /// is skipped if empty.
    ///
    /// [`split`]: #method.split
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
    /// from a forward search, the [`rsplit_terminator`] method can be used.
    ///
    /// [`rsplit_terminator`]: #method.rsplit_terminator
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
    #[inline]
    pub fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P> {
        core_str::StrExt::split_terminator(self, pat)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern and yielded in reverse order.
    ///
    /// The pattern can be a simple `&str`, [`char`], or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// Equivalent to [`split`], except that the trailing substring is
    /// skipped if empty.
    ///
    /// [`split`]: #method.split
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
    /// For iterating from the front, the [`split_terminator`] method can be
    /// used.
    ///
    /// [`split_terminator`]: #method.split_terminator
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
    #[inline]
    pub fn rsplit_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplitTerminator<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplit_terminator(self, pat)
    }

    /// An iterator over substrings of the given string slice, separated by a
    /// pattern, restricted to returning at most `n` items.
    ///
    /// If `n` substrings are returned, the last substring (the `n`th substring)
    /// will contain the remainder of the string.
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
    /// If the pattern allows a reverse search, the [`rsplitn`] method can be
    /// used.
    ///
    /// [`rsplitn`]: #method.rsplitn
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
    #[inline]
    pub fn splitn<'a, P: Pattern<'a>>(&'a self, n: usize, pat: P) -> SplitN<'a, P> {
        core_str::StrExt::splitn(self, n, pat)
    }

    /// An iterator over substrings of this string slice, separated by a
    /// pattern, starting from the end of the string, restricted to returning
    /// at most `n` items.
    ///
    /// If `n` substrings are returned, the last substring (the `n`th substring)
    /// will contain the remainder of the string.
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
    /// For splitting from the front, the [`splitn`] method can be used.
    ///
    /// [`splitn`]: #method.splitn
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
    #[inline]
    pub fn rsplitn<'a, P: Pattern<'a>>(&'a self, n: usize, pat: P) -> RSplitN<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplitn(self, n, pat)
    }

    /// An iterator over the disjoint matches of a pattern within the given string
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
    /// from a forward search, the [`rmatches`] method can be used.
    ///
    /// [`rmatches`]: #method.rmatches
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
    #[inline]
    pub fn matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> Matches<'a, P> {
        core_str::StrExt::matches(self, pat)
    }

    /// An iterator over the disjoint matches of a pattern within this string slice,
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
    /// For iterating from the front, the [`matches`] method can be used.
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
    #[inline]
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
    /// from a forward search, the [`rmatch_indices`] method can be used.
    ///
    /// [`rmatch_indices`]: #method.rmatch_indices
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
    #[inline]
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
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`match_indices`] method can be used.
    ///
    /// [`match_indices`]: #method.match_indices
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
    #[inline]
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
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
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
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English";
    /// assert!(Some('E') == s.trim_left().chars().next());
    ///
    /// let s = "  עברית";
    /// assert!(Some('ע') == s.trim_left().chars().next());
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
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
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
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "English  ";
    /// assert!(Some('h') == s.trim_right().chars().rev().next());
    ///
    /// let s = "עברית  ";
    /// assert!(Some('ת') == s.trim_right().chars().rev().next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_right(&self) -> &str {
        UnicodeStr::trim_right(self)
    }

    /// Returns a string slice with all prefixes and suffixes that match a
    /// pattern repeatedly removed.
    ///
    /// The pattern can be a [`char`] or a closure that determines if a
    /// character matches.
    ///
    /// [`char`]: primitive.char.html
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
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
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
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
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
    /// assert_eq!("1fooX".trim_right_matches(|c| c == '1' || c == 'X'), "1foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_right_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::trim_right_matches(self, pat)
    }

    /// Parses this string slice into another type.
    ///
    /// Because `parse` is so general, it can cause problems with type
    /// inference. As such, `parse` is one of the few times you'll see
    /// the syntax affectionately known as the 'turbofish': `::<>`. This
    /// helps the inference algorithm understand specifically which type
    /// you're trying to parse into.
    ///
    /// `parse` can parse any type that implements the [`FromStr`] trait.
    ///
    /// [`FromStr`]: str/trait.FromStr.html
    ///
    /// # Errors
    ///
    /// Will return [`Err`] if it's not possible to parse this string slice into
    /// the desired type.
    ///
    /// [`Err`]: str/trait.FromStr.html#associatedtype.Err
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// let four: u32 = "4".parse().unwrap();
    ///
    /// assert_eq!(4, four);
    /// ```
    ///
    /// Using the 'turbofish' instead of annotating `four`:
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

    /// Converts a `Box<str>` into a `Box<[u8]>` without copying or allocating.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "this is a string";
    /// let boxed_str = s.to_owned().into_boxed_str();
    /// let boxed_bytes = boxed_str.into_boxed_bytes();
    /// assert_eq!(*boxed_bytes, *s.as_bytes());
    /// ```
    #[stable(feature = "str_box_extras", since = "1.20.0")]
    pub fn into_boxed_bytes(self: Box<str>) -> Box<[u8]> {
        self.into()
    }

    /// Replaces all matches of a pattern with another string.
    ///
    /// `replace` creates a new [`String`], and copies the data from this string slice into it.
    /// While doing so, it attempts to find matches of a pattern. If it finds any, it
    /// replaces them with the replacement string slice.
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
    /// When the pattern doesn't match:
    ///
    /// ```
    /// let s = "this is old";
    /// assert_eq!(s, s.replace("cookie monster", "little lamb"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn replace<'a, P: Pattern<'a>>(&'a self, from: P, to: &str) -> String {
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

    /// Replaces first N matches of a pattern with another string.
    ///
    /// `replacen` creates a new [`String`], and copies the data from this string slice into it.
    /// While doing so, it attempts to find matches of a pattern. If it finds any, it
    /// replaces them with the replacement string slice at most `count` times.
    ///
    /// [`String`]: string/struct.String.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "foo foo 123 foo";
    /// assert_eq!("new new 123 foo", s.replacen("foo", "new", 2));
    /// assert_eq!("faa fao 123 foo", s.replacen('o', "a", 3));
    /// assert_eq!("foo foo new23 foo", s.replacen(char::is_numeric, "new", 1));
    /// ```
    ///
    /// When the pattern doesn't match:
    ///
    /// ```
    /// let s = "this is old";
    /// assert_eq!(s, s.replacen("cookie monster", "little lamb", 10));
    /// ```
    #[stable(feature = "str_replacen", since = "1.16.0")]
    pub fn replacen<'a, P: Pattern<'a>>(&'a self, pat: P, to: &str, count: usize) -> String {
        // Hope to reduce the times of re-allocation
        let mut result = String::with_capacity(32);
        let mut last_end = 0;
        for (start, part) in self.match_indices(pat).take(count) {
            result.push_str(unsafe { self.slice_unchecked(last_end, start) });
            result.push_str(to);
            last_end = start + part.len();
        }
        result.push_str(unsafe { self.slice_unchecked(last_end, self.len()) });
        result
    }

    /// Returns the lowercase equivalent of this string slice, as a new [`String`].
    ///
    /// 'Lowercase' is defined according to the terms of the Unicode Derived Core Property
    /// `Lowercase`.
    ///
    /// Since some characters can expand into multiple characters when changing
    /// the case, this function returns a [`String`] instead of modifying the
    /// parameter in-place.
    ///
    /// [`String`]: string/struct.String.html
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
                // so hard-code it rather than have a generic "condition" mechanism.
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
            to.push_str(if is_word_final { "ς" } else { "σ" });
        }

        fn case_ignoreable_then_cased<I: Iterator<Item = char>>(iter: I) -> bool {
            use std_unicode::derived_property::{Cased, Case_Ignorable};
            match iter.skip_while(|&c| Case_Ignorable(c)).next() {
                Some(c) => Cased(c),
                None => false,
            }
        }
    }

    /// Returns the uppercase equivalent of this string slice, as a new [`String`].
    ///
    /// 'Uppercase' is defined according to the terms of the Unicode Derived Core Property
    /// `Uppercase`.
    ///
    /// Since some characters can expand into multiple characters when changing
    /// the case, this function returns a [`String`] instead of modifying the
    /// parameter in-place.
    ///
    /// [`String`]: string/struct.String.html
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

    /// Escapes each char in `s` with [`char::escape_debug`].
    ///
    /// [`char::escape_debug`]: primitive.char.html#method.escape_debug
    #[unstable(feature = "str_escape",
               reason = "return type may change to be an iterator",
               issue = "27791")]
    pub fn escape_debug(&self) -> String {
        self.chars().flat_map(|c| c.escape_debug()).collect()
    }

    /// Escapes each char in `s` with [`char::escape_default`].
    ///
    /// [`char::escape_default`]: primitive.char.html#method.escape_default
    #[unstable(feature = "str_escape",
               reason = "return type may change to be an iterator",
               issue = "27791")]
    pub fn escape_default(&self) -> String {
        self.chars().flat_map(|c| c.escape_default()).collect()
    }

    /// Escapes each char in `s` with [`char::escape_unicode`].
    ///
    /// [`char::escape_unicode`]: primitive.char.html#method.escape_unicode
    #[unstable(feature = "str_escape",
               reason = "return type may change to be an iterator",
               issue = "27791")]
    pub fn escape_unicode(&self) -> String {
        self.chars().flat_map(|c| c.escape_unicode()).collect()
    }

    /// Converts a [`Box<str>`] into a [`String`] without copying or allocating.
    ///
    /// [`String`]: string/struct.String.html
    /// [`Box<str>`]: boxed/struct.Box.html
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
        let slice = Box::<[u8]>::from(self);
        unsafe { String::from_utf8_unchecked(slice.into_vec()) }
    }

    /// Create a [`String`] by repeating a string `n` times.
    ///
    /// [`String`]: string/struct.String.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!("abc".repeat(4), String::from("abcabcabcabc"));
    /// ```
    #[stable(feature = "repeat_str", since = "1.16.0")]
    pub fn repeat(&self, n: usize) -> String {
        let mut s = String::with_capacity(self.len() * n);
        s.extend((0..n).map(|_| self));
        s
    }

    /// Checks if all characters in this string are within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = "hello!\n";
    /// let non_ascii = "Grüße, Jürgen ❤";
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.21.0")]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        // We can treat each byte as character here: all multibyte characters
        // start with a byte that is not in the ascii range, so we will stop
        // there already.
        self.bytes().all(|b| b.is_ascii())
    }

    /// Returns a copy of this string where each character is mapped to its
    /// ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// To uppercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Grüße, Jürgen ❤";
    ///
    /// assert_eq!("GRüßE, JüRGEN ❤", s.to_ascii_uppercase());
    /// ```
    ///
    /// [`make_ascii_uppercase`]: #method.make_ascii_uppercase
    /// [`to_uppercase`]: #method.to_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.21.0")]
    #[inline]
    #[cfg(not(stage0))]
    pub fn to_ascii_uppercase(&self) -> String {
        let mut bytes = self.as_bytes().to_vec();
        bytes.make_ascii_uppercase();
        // make_ascii_uppercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(bytes) }
    }

    /// Returns a copy of this string where each character is mapped to its
    /// ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// To lowercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Grüße, Jürgen ❤";
    ///
    /// assert_eq!("grüße, jürgen ❤", s.to_ascii_lowercase());
    /// ```
    ///
    /// [`make_ascii_lowercase`]: #method.make_ascii_lowercase
    /// [`to_lowercase`]: #method.to_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.21.0")]
    #[inline]
    #[cfg(not(stage0))]
    pub fn to_ascii_lowercase(&self) -> String {
        let mut bytes = self.as_bytes().to_vec();
        bytes.make_ascii_lowercase();
        // make_ascii_lowercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(bytes) }
    }

    /// Checks that two strings are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!("Ferris".eq_ignore_ascii_case("FERRIS"));
    /// assert!("Ferrös".eq_ignore_ascii_case("FERRöS"));
    /// assert!(!"Ferrös".eq_ignore_ascii_case("FERRÖS"));
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.21.0")]
    #[inline]
    #[cfg(not(stage0))]
    pub fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    }

    /// Converts this string to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.21.0")]
    #[cfg(not(stage0))]
    pub fn make_ascii_uppercase(&mut self) {
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_uppercase()
    }

    /// Converts this string to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.21.0")]
    #[cfg(not(stage0))]
    pub fn make_ascii_lowercase(&mut self) {
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_lowercase()
    }

    /// Checks if all characters of this string are ASCII alphabetic
    /// characters:
    ///
    /// - U+0041 'A' ... U+005A 'Z', or
    /// - U+0061 'a' ... U+007A 'z'.
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_alphabetic(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_alphabetic())
    }

    /// Checks if all characters of this string are ASCII uppercase characters:
    /// U+0041 'A' ... U+005A 'Z'.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    ///
    /// // Only ascii uppercase characters
    /// assert!("HELLO".is_ascii_uppercase());
    ///
    /// // While all characters are ascii, 'y' and 'e' are not uppercase
    /// assert!(!"Bye".is_ascii_uppercase());
    ///
    /// // While all characters are uppercase, 'Ü' is not ascii
    /// assert!(!"TSCHÜSS".is_ascii_uppercase());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_uppercase(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_uppercase())
    }

    /// Checks if all characters of this string are ASCII lowercase characters:
    /// U+0061 'a' ... U+007A 'z'.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(ascii_ctype)]
    ///
    /// // Only ascii uppercase characters
    /// assert!("hello".is_ascii_lowercase());
    ///
    /// // While all characters are ascii, 'B' is not lowercase
    /// assert!(!"Bye".is_ascii_lowercase());
    ///
    /// // While all characters are lowercase, 'Ü' is not ascii
    /// assert!(!"tschüss".is_ascii_lowercase());
    /// ```
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_lowercase(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_lowercase())
    }

    /// Checks if all characters of this string are ASCII alphanumeric
    /// characters:
    ///
    /// - U+0041 'A' ... U+005A 'Z', or
    /// - U+0061 'a' ... U+007A 'z', or
    /// - U+0030 '0' ... U+0039 '9'.
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_alphanumeric(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_alphanumeric())
    }

    /// Checks if all characters of this string are ASCII decimal digit:
    /// U+0030 '0' ... U+0039 '9'.
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_digit(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_digit())
    }

    /// Checks if all characters of this string are ASCII hexadecimal digits:
    ///
    /// - U+0030 '0' ... U+0039 '9', or
    /// - U+0041 'A' ... U+0046 'F', or
    /// - U+0061 'a' ... U+0066 'f'.
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_hexdigit(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_hexdigit())
    }

    /// Checks if all characters of this string are ASCII punctuation
    /// characters:
    ///
    /// - U+0021 ... U+002F `! " # $ % & ' ( ) * + , - . /`, or
    /// - U+003A ... U+0040 `: ; < = > ? @`, or
    /// - U+005B ... U+0060 ``[ \ ] ^ _ ` ``, or
    /// - U+007B ... U+007E `{ | } ~`
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_punctuation(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_punctuation())
    }

    /// Checks if all characters of this string are ASCII graphic characters:
    /// U+0021 '@' ... U+007E '~'.
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_graphic(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_graphic())
    }

    /// Checks if all characters of this string are ASCII whitespace characters:
    /// U+0020 SPACE, U+0009 HORIZONTAL TAB, U+000A LINE FEED,
    /// U+000C FORM FEED, or U+000D CARRIAGE RETURN.
    ///
    /// Rust uses the WhatWG Infra Standard's [definition of ASCII
    /// whitespace][infra-aw]. There are several other definitions in
    /// wide use. For instance, [the POSIX locale][pct] includes
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
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_whitespace(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_whitespace())
    }

    /// Checks if all characters of this string are ASCII control characters:
    ///
    /// - U+0000 NUL ... U+001F UNIT SEPARATOR, or
    /// - U+007F DELETE.
    ///
    /// Note that most ASCII whitespace characters are control
    /// characters, but SPACE is not.
    #[unstable(feature = "ascii_ctype", issue = "39658")]
    #[inline]
    pub fn is_ascii_control(&self) -> bool {
        self.bytes().all(|b| b.is_ascii_control())
    }
}

/// Converts a boxed slice of bytes to a boxed string slice without checking
/// that the string contains valid UTF-8.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let smile_utf8 = Box::new([226, 152, 186]);
/// let smile = unsafe { std::str::from_boxed_utf8_unchecked(smile_utf8) };
///
/// assert_eq!("☺", &*smile);
/// ```
#[stable(feature = "str_box_extras", since = "1.20.0")]
pub unsafe fn from_boxed_utf8_unchecked(v: Box<[u8]>) -> Box<str> {
    Box::from_raw(Box::into_raw(v) as *mut str)
}
