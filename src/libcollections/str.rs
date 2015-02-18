// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

//! Unicode string manipulation (`str` type)
//!
//! # Basic Usage
//!
//! Rust's string type is one of the core primitive types of the language. While
//! represented by the name `str`, the name `str` is not actually a valid type in
//! Rust. Each string must also be decorated with a pointer. `String` is used
//! for an owned string, so there is only one commonly-used `str` type in Rust:
//! `&str`.
//!
//! `&str` is the borrowed string type. This type of string can only be created
//! from other strings, unless it is a static string (see below). As the word
//! "borrowed" implies, this type of string is owned elsewhere, and this string
//! cannot be moved out of.
//!
//! As an example, here's some code that uses a string.
//!
//! ```rust
//! fn main() {
//!     let borrowed_string = "This string is borrowed with the 'static lifetime";
//! }
//! ```
//!
//! From the example above, you can guess that Rust's string literals have the
//! `'static` lifetime. This is akin to C's concept of a static string.
//! More precisely, string literals are immutable views with a 'static lifetime
//! (otherwise known as the lifetime of the entire program), and thus have the
//! type `&'static str`.
//!
//! # Representation
//!
//! Rust's string type, `str`, is a sequence of Unicode scalar values encoded as a
//! stream of UTF-8 bytes. All [strings](../../reference.html#literals) are
//! guaranteed to be validly encoded UTF-8 sequences. Additionally, strings are
//! not null-terminated and can thus contain null bytes.
//!
//! The actual representation of strings have direct mappings to slices: `&str`
//! is the same as `&[u8]`.

#![doc(primitive = "str")]
#![stable(feature = "rust1", since = "1.0.0")]

use self::RecompositionState::*;
use self::DecompositionType::*;

use core::char::CharExt;
use core::clone::Clone;
use core::iter::AdditiveIterator;
use core::iter::{Iterator, IteratorExt};
use core::ops::Index;
use core::ops::RangeFull;
use core::option::Option::{self, Some, None};
use core::result::Result;
use core::slice::AsSlice;
use core::str as core_str;
use unicode::str::{UnicodeStr, Utf16Encoder};

use vec_deque::VecDeque;
use borrow::{Borrow, ToOwned};
use slice::SliceExt;
use string::String;
use unicode;
use vec::Vec;
use slice::SliceConcatExt;

pub use core::str::{FromStr, Utf8Error, Str};
pub use core::str::{Lines, LinesAny, MatchIndices, SplitStr, CharRange};
pub use core::str::{Split, SplitTerminator};
pub use core::str::{SplitN, RSplitN};
pub use core::str::{from_utf8, CharEq, Chars, CharIndices, Bytes};
pub use core::str::{from_utf8_unchecked, from_c_str, ParseBoolError};
pub use unicode::str::{Words, Graphemes, GraphemeIndices};

/*
Section: Creating a string
*/

impl<S: Str> SliceConcatExt<str, String> for [S] {
    fn concat(&self) -> String {
        let s = self.as_slice();

        if s.is_empty() {
            return String::new();
        }

        // `len` calculation may overflow but push_str will check boundaries
        let len = s.iter().map(|s| s.as_slice().len()).sum();
        let mut result = String::with_capacity(len);

        for s in s {
            result.push_str(s.as_slice())
        }

        result
    }

    fn connect(&self, sep: &str) -> String {
        let s = self.as_slice();

        if s.is_empty() {
            return String::new();
        }

        // concat is faster
        if sep.is_empty() {
            return s.concat();
        }

        // this is wrong without the guarantee that `self` is non-empty
        // `len` calculation may overflow but push_str but will check boundaries
        let len = sep.len() * (s.len() - 1)
            + s.iter().map(|s| s.as_slice().len()).sum();
        let mut result = String::with_capacity(len);
        let mut first = true;

        for s in s {
            if first {
                first = false;
            } else {
                result.push_str(sep);
            }
            result.push_str(s.as_slice());
        }
        result
    }
}

/*
Section: Iterators
*/

// Helper functions used for Unicode normalization
fn canonical_sort(comb: &mut [(char, u8)]) {
    let len = comb.len();
    for i in 0..len {
        let mut swapped = false;
        for j in 1..len-i {
            let class_a = comb[j-1].1;
            let class_b = comb[j].1;
            if class_a != 0 && class_b != 0 && class_a > class_b {
                comb.swap(j-1, j);
                swapped = true;
            }
        }
        if !swapped { break; }
    }
}

#[derive(Clone)]
enum DecompositionType {
    Canonical,
    Compatible
}

/// External iterator for a string's decomposition's characters.
/// Use with the `std::iter` module.
#[derive(Clone)]
#[unstable(feature = "collections")]
pub struct Decompositions<'a> {
    kind: DecompositionType,
    iter: Chars<'a>,
    buffer: Vec<(char, u8)>,
    sorted: bool
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Decompositions<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        match self.buffer.first() {
            Some(&(c, 0)) => {
                self.sorted = false;
                self.buffer.remove(0);
                return Some(c);
            }
            Some(&(c, _)) if self.sorted => {
                self.buffer.remove(0);
                return Some(c);
            }
            _ => self.sorted = false
        }

        if !self.sorted {
            for ch in self.iter.by_ref() {
                let buffer = &mut self.buffer;
                let sorted = &mut self.sorted;
                {
                    let callback = |d| {
                        let class =
                            unicode::char::canonical_combining_class(d);
                        if class == 0 && !*sorted {
                            canonical_sort(buffer);
                            *sorted = true;
                        }
                        buffer.push((d, class));
                    };
                    match self.kind {
                        Canonical => {
                            unicode::char::decompose_canonical(ch, callback)
                        }
                        Compatible => {
                            unicode::char::decompose_compatible(ch, callback)
                        }
                    }
                }
                if *sorted {
                    break
                }
            }
        }

        if !self.sorted {
            canonical_sort(&mut self.buffer);
            self.sorted = true;
        }

        if self.buffer.is_empty() {
            None
        } else {
            match self.buffer.remove(0) {
                (c, 0) => {
                    self.sorted = false;
                    Some(c)
                }
                (c, _) => Some(c),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, _) = self.iter.size_hint();
        (lower, None)
    }
}

#[derive(Clone)]
enum RecompositionState {
    Composing,
    Purging,
    Finished
}

/// External iterator for a string's recomposition's characters.
/// Use with the `std::iter` module.
#[derive(Clone)]
#[unstable(feature = "collections")]
pub struct Recompositions<'a> {
    iter: Decompositions<'a>,
    state: RecompositionState,
    buffer: VecDeque<char>,
    composee: Option<char>,
    last_ccc: Option<u8>
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Recompositions<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        loop {
            match self.state {
                Composing => {
                    for ch in self.iter.by_ref() {
                        let ch_class = unicode::char::canonical_combining_class(ch);
                        if self.composee.is_none() {
                            if ch_class != 0 {
                                return Some(ch);
                            }
                            self.composee = Some(ch);
                            continue;
                        }
                        let k = self.composee.clone().unwrap();

                        match self.last_ccc {
                            None => {
                                match unicode::char::compose(k, ch) {
                                    Some(r) => {
                                        self.composee = Some(r);
                                        continue;
                                    }
                                    None => {
                                        if ch_class == 0 {
                                            self.composee = Some(ch);
                                            return Some(k);
                                        }
                                        self.buffer.push_back(ch);
                                        self.last_ccc = Some(ch_class);
                                    }
                                }
                            }
                            Some(l_class) => {
                                if l_class >= ch_class {
                                    // `ch` is blocked from `composee`
                                    if ch_class == 0 {
                                        self.composee = Some(ch);
                                        self.last_ccc = None;
                                        self.state = Purging;
                                        return Some(k);
                                    }
                                    self.buffer.push_back(ch);
                                    self.last_ccc = Some(ch_class);
                                    continue;
                                }
                                match unicode::char::compose(k, ch) {
                                    Some(r) => {
                                        self.composee = Some(r);
                                        continue;
                                    }
                                    None => {
                                        self.buffer.push_back(ch);
                                        self.last_ccc = Some(ch_class);
                                    }
                                }
                            }
                        }
                    }
                    self.state = Finished;
                    if self.composee.is_some() {
                        return self.composee.take();
                    }
                }
                Purging => {
                    match self.buffer.pop_front() {
                        None => self.state = Composing,
                        s => return s
                    }
                }
                Finished => {
                    match self.buffer.pop_front() {
                        None => return self.composee.take(),
                        s => return s
                    }
                }
            }
        }
    }
}

/// External iterator for a string's UTF16 codeunits.
/// Use with the `std::iter` module.
#[derive(Clone)]
#[unstable(feature = "collections")]
pub struct Utf16Units<'a> {
    encoder: Utf16Encoder<Chars<'a>>
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Utf16Units<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> { self.encoder.next() }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.encoder.size_hint() }
}

/*
Section: Misc
*/

// Return the initial codepoint accumulator for the first byte.
// The first byte is special, only want bottom 5 bits for width 2, 4 bits
// for width 3, and 3 bits for width 4
macro_rules! utf8_first_byte {
    ($byte:expr, $width:expr) => (($byte & (0x7F >> $width)) as u32)
}

// return the value of $ch updated with continuation byte $byte
macro_rules! utf8_acc_cont_byte {
    ($ch:expr, $byte:expr) => (($ch << 6) | ($byte & 63u8) as u32)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<str> for String {
    fn borrow(&self) -> &str { &self[..] }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for str {
    type Owned = String;
    fn to_owned(&self) -> String {
        unsafe {
            String::from_utf8_unchecked(self.as_bytes().to_owned())
        }
    }
}

/*
Section: CowString
*/

/*
Section: Trait implementations
*/

/// Any string that can be represented as a slice.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait StrExt: Index<RangeFull, Output = str> {
    /// Escapes each char in `s` with `char::escape_default`.
    #[unstable(feature = "collections",
               reason = "return type may change to be an iterator")]
    fn escape_default(&self) -> String {
        self.chars().flat_map(|c| c.escape_default()).collect()
    }

    /// Escapes each char in `s` with `char::escape_unicode`.
    #[unstable(feature = "collections",
               reason = "return type may change to be an iterator")]
    fn escape_unicode(&self) -> String {
        self.chars().flat_map(|c| c.escape_unicode()).collect()
    }

    /// Replaces all occurrences of one string with another.
    ///
    /// # Arguments
    ///
    /// * `from` - The string to replace
    /// * `to` - The replacement string
    ///
    /// # Return value
    ///
    /// The original string with all occurrences of `from` replaced with `to`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let s = "this is old";
    ///
    /// assert_eq!(s.replace("old", "new"), "this is new");
    ///
    /// // not found, so no change.
    /// assert_eq!(s.replace("cookie monster", "little lamb"), s);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn replace(&self, from: &str, to: &str) -> String {
        let mut result = String::new();
        let mut last_end = 0;
        for (start, end) in self.match_indices(from) {
            result.push_str(unsafe { self.slice_unchecked(last_end, start) });
            result.push_str(to);
            last_end = end;
        }
        result.push_str(unsafe { self.slice_unchecked(last_end, self.len()) });
        result
    }

    /// Returns an iterator over the string in Unicode Normalization Form D
    /// (canonical decomposition).
    #[inline]
    #[unstable(feature = "collections",
               reason = "this functionality may be moved to libunicode")]
    fn nfd_chars(&self) -> Decompositions {
        Decompositions {
            iter: self[..].chars(),
            buffer: Vec::new(),
            sorted: false,
            kind: Canonical
        }
    }

    /// Returns an iterator over the string in Unicode Normalization Form KD
    /// (compatibility decomposition).
    #[inline]
    #[unstable(feature = "collections",
               reason = "this functionality may be moved to libunicode")]
    fn nfkd_chars(&self) -> Decompositions {
        Decompositions {
            iter: self[..].chars(),
            buffer: Vec::new(),
            sorted: false,
            kind: Compatible
        }
    }

    /// An Iterator over the string in Unicode Normalization Form C
    /// (canonical decomposition followed by canonical composition).
    #[inline]
    #[unstable(feature = "collections",
               reason = "this functionality may be moved to libunicode")]
    fn nfc_chars(&self) -> Recompositions {
        Recompositions {
            iter: self.nfd_chars(),
            state: Composing,
            buffer: VecDeque::new(),
            composee: None,
            last_ccc: None
        }
    }

    /// An Iterator over the string in Unicode Normalization Form KC
    /// (compatibility decomposition followed by canonical composition).
    #[inline]
    #[unstable(feature = "collections",
               reason = "this functionality may be moved to libunicode")]
    fn nfkc_chars(&self) -> Recompositions {
        Recompositions {
            iter: self.nfkd_chars(),
            state: Composing,
            buffer: VecDeque::new(),
            composee: None,
            last_ccc: None
        }
    }

    /// Returns true if a string contains a string pattern.
    ///
    /// # Arguments
    ///
    /// - pat - The string pattern to look for
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!("bananas".contains("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn contains(&self, pat: &str) -> bool {
        core_str::StrExt::contains(&self[..], pat)
    }

    /// Returns true if a string contains a char pattern.
    ///
    /// # Arguments
    ///
    /// - pat - The char pattern to look for
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!("hello".contains_char('e'));
    /// ```
    #[unstable(feature = "collections",
               reason = "might get removed in favour of a more generic contains()")]
    fn contains_char<P: CharEq>(&self, pat: P) -> bool {
        core_str::StrExt::contains_char(&self[..], pat)
    }

    /// An iterator over the characters of `self`. Note, this iterates
    /// over Unicode code-points, not Unicode graphemes.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<char> = "abc åäö".chars().collect();
    /// assert_eq!(v, vec!['a', 'b', 'c', ' ', 'å', 'ä', 'ö']);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn chars(&self) -> Chars {
        core_str::StrExt::chars(&self[..])
    }

    /// An iterator over the bytes of `self`
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<u8> = "bors".bytes().collect();
    /// assert_eq!(v, b"bors".to_vec());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn bytes(&self) -> Bytes {
        core_str::StrExt::bytes(&self[..])
    }

    /// An iterator over the characters of `self` and their byte offsets.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn char_indices(&self) -> CharIndices {
        core_str::StrExt::char_indices(&self[..])
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by the pattern `pat`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "Mary had a little lamb".split(' ').collect();
    /// assert_eq!(v, vec!["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".split(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, vec!["abc", "def", "ghi"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').collect();
    /// assert_eq!(v, vec!["lion", "", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "".split('X').collect();
    /// assert_eq!(v, vec![""]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn split<P: CharEq>(&self, pat: P) -> Split<P> {
        core_str::StrExt::split(&self[..], pat)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by the pattern `pat`, restricted to splitting at most `count`
    /// times.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "Mary had a little lambda".splitn(2, ' ').collect();
    /// assert_eq!(v, vec!["Mary", "had", "a little lambda"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".splitn(1, |c: char| c.is_numeric()).collect();
    /// assert_eq!(v, vec!["abc", "def2ghi"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".splitn(2, 'X').collect();
    /// assert_eq!(v, vec!["lion", "", "tigerXleopard"]);
    ///
    /// let v: Vec<&str> = "abcXdef".splitn(0, 'X').collect();
    /// assert_eq!(v, vec!["abcXdef"]);
    ///
    /// let v: Vec<&str> = "".splitn(1, 'X').collect();
    /// assert_eq!(v, vec![""]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn splitn<P: CharEq>(&self, count: usize, pat: P) -> SplitN<P> {
        core_str::StrExt::splitn(&self[..], count, pat)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by the pattern `pat`.
    ///
    /// Equivalent to `split`, except that the trailing substring
    /// is skipped if empty (terminator semantics).
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "A.B.".split_terminator('.').collect();
    /// assert_eq!(v, vec!["A", "B"]);
    ///
    /// let v: Vec<&str> = "A..B..".split_terminator('.').collect();
    /// assert_eq!(v, vec!["A", "", "B", ""]);
    ///
    /// let v: Vec<&str> = "Mary had a little lamb".split(' ').rev().collect();
    /// assert_eq!(v, vec!["lamb", "little", "a", "had", "Mary"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".split(|c: char| c.is_numeric()).rev().collect();
    /// assert_eq!(v, vec!["ghi", "def", "abc"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').rev().collect();
    /// assert_eq!(v, vec!["leopard", "tiger", "", "lion"]);
    /// ```
    #[unstable(feature = "collections", reason = "might get removed")]
    fn split_terminator<P: CharEq>(&self, pat: P) -> SplitTerminator<P> {
        core_str::StrExt::split_terminator(&self[..], pat)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by the pattern `pat`, starting from the end of the string.
    /// Restricted to splitting at most `count` times.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "Mary had a little lamb".rsplitn(2, ' ').collect();
    /// assert_eq!(v, vec!["lamb", "little", "Mary had a"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".rsplitn(1, |c: char| c.is_numeric()).collect();
    /// assert_eq!(v, vec!["ghi", "abc1def"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplitn(2, 'X').collect();
    /// assert_eq!(v, vec!["leopard", "tiger", "lionX"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rsplitn<P: CharEq>(&self, count: usize, pat: P) -> RSplitN<P> {
        core_str::StrExt::rsplitn(&self[..], count, pat)
    }

    /// An iterator over the start and end indices of the disjoint
    /// matches of the pattern `pat` within `self`.
    ///
    /// That is, each returned value `(start, end)` satisfies
    /// `self.slice(start, end) == sep`. For matches of `sep` within
    /// `self` that overlap, only the indices corresponding to the
    /// first match are returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<(usize, usize)> = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, vec![(0,3), (6,9), (12,15)]);
    ///
    /// let v: Vec<(usize, usize)> = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, vec![(1,4), (4,7)]);
    ///
    /// let v: Vec<(usize, usize)> = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, vec![(0, 3)]); // only the first `aba`
    /// ```
    #[unstable(feature = "collections",
               reason = "might have its iterator type changed")]
    fn match_indices<'a>(&'a self, pat: &'a str) -> MatchIndices<'a> {
        core_str::StrExt::match_indices(&self[..], pat)
    }

    /// An iterator over the substrings of `self` separated by the pattern `sep`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "abcXXXabcYYYabc".split_str("abc").collect();
    /// assert_eq!(v, vec!["", "XXX", "YYY", ""]);
    ///
    /// let v: Vec<&str> = "1abcabc2".split_str("abc").collect();
    /// assert_eq!(v, vec!["1", "", "2"]);
    /// ```
    #[unstable(feature = "collections",
               reason = "might get removed in the future in favor of a more generic split()")]
    fn split_str<'a>(&'a self, pat: &'a str) -> SplitStr<'a> {
        core_str::StrExt::split_str(&self[..], pat)
    }

    /// An iterator over the lines of a string (subsequences separated
    /// by `\n`). This does not include the empty string after a
    /// trailing `\n`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let four_lines = "foo\nbar\n\nbaz\n";
    /// let v: Vec<&str> = four_lines.lines().collect();
    /// assert_eq!(v, vec!["foo", "bar", "", "baz"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn lines(&self) -> Lines {
        core_str::StrExt::lines(&self[..])
    }

    /// An iterator over the lines of a string, separated by either
    /// `\n` or `\r\n`. As with `.lines()`, this does not include an
    /// empty trailing line.
    ///
    /// # Example
    ///
    /// ```rust
    /// let four_lines = "foo\r\nbar\n\r\nbaz\n";
    /// let v: Vec<&str> = four_lines.lines_any().collect();
    /// assert_eq!(v, vec!["foo", "bar", "", "baz"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn lines_any(&self) -> LinesAny {
        core_str::StrExt::lines_any(&self[..])
    }

    /// Deprecated: use `s[a .. b]` instead.
    #[unstable(feature = "collections",
               reason = "use slice notation [a..b] instead")]
    #[deprecated(since = "1.0.0", reason = "use slice notation [a..b] instead")]
    fn slice(&self, begin: usize, end: usize) -> &str;

    /// Deprecated: use `s[a..]` instead.
    #[unstable(feature = "collections",
               reason = "use slice notation [a..b] instead")]
    #[deprecated(since = "1.0.0", reason = "use slice notation [a..] instead")]
    fn slice_from(&self, begin: usize) -> &str;

    /// Deprecated: use `s[..a]` instead.
    #[unstable(feature = "collections",
               reason = "use slice notation [a..b] instead")]
    #[deprecated(since = "1.0.0", reason = "use slice notation [..a] instead")]
    fn slice_to(&self, end: usize) -> &str;

    /// Returns a slice of the string from the character range
    /// [`begin`..`end`).
    ///
    /// That is, start at the `begin`-th code point of the string and
    /// continue to the `end`-th code point. This does not detect or
    /// handle edge cases such as leaving a combining character as the
    /// first code point of the string.
    ///
    /// Due to the design of UTF-8, this operation is `O(end)`.
    /// See `slice`, `slice_to` and `slice_from` for `O(1)`
    /// variants that use byte indices rather than code point
    /// indices.
    ///
    /// Panics if `begin` > `end` or the either `begin` or `end` are
    /// beyond the last character of the string.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// assert_eq!(s.slice_chars(0, 4), "Löwe");
    /// assert_eq!(s.slice_chars(5, 7), "老虎");
    /// ```
    #[unstable(feature = "collections",
               reason = "may have yet to prove its worth")]
    fn slice_chars(&self, begin: usize, end: usize) -> &str {
        core_str::StrExt::slice_chars(&self[..], begin, end)
    }

    /// Takes a bytewise (not UTF-8) slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// Caller must check both UTF-8 character boundaries and the boundaries of
    /// the entire slice as well.
    #[stable(feature = "rust1", since = "1.0.0")]
    unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        core_str::StrExt::slice_unchecked(&self[..], begin, end)
    }

    /// Returns true if the pattern `pat` is a prefix of the string.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!("banana".starts_with("ba"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn starts_with(&self, pat: &str) -> bool {
        core_str::StrExt::starts_with(&self[..], pat)
    }

    /// Returns true if the pattern `pat` is a suffix of the string.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!("banana".ends_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ends_with(&self, pat: &str) -> bool {
        core_str::StrExt::ends_with(&self[..], pat)
    }

    /// Returns a string with all pre- and suffixes that match
    /// the pattern `pat` repeatedly removed.
    ///
    /// # Arguments
    ///
    /// * pat - a string pattern
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_matches(x), "foo1bar");
    /// assert_eq!("123foo1bar123".trim_matches(|c: char| c.is_numeric()), "foo1bar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trim_matches<P: CharEq>(&self, pat: P) -> &str {
        core_str::StrExt::trim_matches(&self[..], pat)
    }

    /// Returns a string with all prefixes that match
    /// the pattern `pat` repeatedly removed.
    ///
    /// # Arguments
    ///
    /// * pat - a string pattern
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_left_matches(x), "foo1bar12");
    /// assert_eq!("123foo1bar123".trim_left_matches(|c: char| c.is_numeric()), "foo1bar123");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trim_left_matches<P: CharEq>(&self, pat: P) -> &str {
        core_str::StrExt::trim_left_matches(&self[..], pat)
    }

    /// Returns a string with all suffixes that match
    /// the pattern `pat` repeatedly removed.
    ///
    /// # Arguments
    ///
    /// * pat - a string pattern
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_right_matches(x), "12foo1bar");
    /// assert_eq!("123foo1bar123".trim_right_matches(|c: char| c.is_numeric()), "123foo1bar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trim_right_matches<P: CharEq>(&self, pat: P) -> &str {
        core_str::StrExt::trim_right_matches(&self[..], pat)
    }

    /// Check that `index`-th byte lies at the start and/or end of a
    /// UTF-8 code point sequence.
    ///
    /// The start and end of the string (when `index == self.len()`)
    /// are considered to be boundaries.
    ///
    /// Panics if `index` is greater than `self.len()`.
    ///
    /// # Example
    ///
    /// ```rust
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
    #[unstable(feature = "collections",
               reason = "naming is uncertain with container conventions")]
    fn is_char_boundary(&self, index: usize) -> bool {
        core_str::StrExt::is_char_boundary(&self[..], index)
    }

    /// Pluck a character out of a string and return the index of the next
    /// character.
    ///
    /// This function can be used to iterate over the Unicode characters of a
    /// string.
    ///
    /// # Example
    ///
    /// This example manually iterates through the characters of a
    /// string; this should normally be done by `.chars()` or
    /// `.char_indices`.
    ///
    /// ```rust
    /// use std::str::CharRange;
    ///
    /// let s = "中华Việt Nam";
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
    /// 8: ệ
    /// 11: t
    /// 12:
    /// 13: N
    /// 14: a
    /// 15: m
    /// ```
    ///
    /// # Arguments
    ///
    /// * s - The string
    /// * i - The byte offset of the char to extract
    ///
    /// # Return value
    ///
    /// A record {ch: char, next: usize} containing the char value and the byte
    /// index of the next Unicode character.
    ///
    /// # Panics
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 character.
    #[unstable(feature = "collections",
               reason = "naming is uncertain with container conventions")]
    fn char_range_at(&self, start: usize) -> CharRange {
        core_str::StrExt::char_range_at(&self[..], start)
    }

    /// Given a byte position and a str, return the previous char and its position.
    ///
    /// This function can be used to iterate over a Unicode string in reverse.
    ///
    /// Returns 0 for next index if called on start index 0.
    ///
    /// # Panics
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 character.
    #[unstable(feature = "collections",
               reason = "naming is uncertain with container conventions")]
    fn char_range_at_reverse(&self, start: usize) -> CharRange {
        core_str::StrExt::char_range_at_reverse(&self[..], start)
    }

    /// Plucks the character starting at the `i`th byte of a string.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "abπc";
    /// assert_eq!(s.char_at(1), 'b');
    /// assert_eq!(s.char_at(2), 'π');
    /// assert_eq!(s.char_at(4), 'c');
    /// ```
    ///
    /// # Panics
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 character.
    #[unstable(feature = "collections",
               reason = "naming is uncertain with container conventions")]
    fn char_at(&self, i: usize) -> char {
        core_str::StrExt::char_at(&self[..], i)
    }

    /// Plucks the character ending at the `i`th byte of a string.
    ///
    /// # Panics
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 character.
    #[unstable(feature = "collections",
               reason = "naming is uncertain with container conventions")]
    fn char_at_reverse(&self, i: usize) -> char {
        core_str::StrExt::char_at_reverse(&self[..], i)
    }

    /// Work with the byte buffer of a string as a byte slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("bors".as_bytes(), b"bors");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_bytes(&self) -> &[u8] {
        core_str::StrExt::as_bytes(&self[..])
    }

    /// Returns the byte index of the first character of `self` that
    /// matches the pattern `pat`.
    ///
    /// # Return value
    ///
    /// `Some` containing the byte index of the last matching character
    /// or `None` if there is no match
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find('L'), Some(0));
    /// assert_eq!(s.find('é'), Some(14));
    ///
    /// // the first space
    /// assert_eq!(s.find(|c: char| c.is_whitespace()), Some(5));
    ///
    /// // neither are found
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!(s.find(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn find<P: CharEq>(&self, pat: P) -> Option<usize> {
        core_str::StrExt::find(&self[..], pat)
    }

    /// Returns the byte index of the last character of `self` that
    /// matches the pattern `pat`.
    ///
    /// # Return value
    ///
    /// `Some` containing the byte index of the last matching character
    /// or `None` if there is no match.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind('L'), Some(13));
    /// assert_eq!(s.rfind('é'), Some(14));
    ///
    /// // the second space
    /// assert_eq!(s.rfind(|c: char| c.is_whitespace()), Some(12));
    ///
    /// // searches for an occurrence of either `1` or `2`, but neither are found
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!(s.rfind(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn rfind<P: CharEq>(&self, pat: P) -> Option<usize> {
        core_str::StrExt::rfind(&self[..], pat)
    }

    /// Returns the byte index of the first matching substring
    ///
    /// # Arguments
    ///
    /// * `needle` - The string to search for
    ///
    /// # Return value
    ///
    /// `Some` containing the byte index of the first matching substring
    /// or `None` if there is no match.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find_str("老虎 L"), Some(6));
    /// assert_eq!(s.find_str("muffin man"), None);
    /// ```
    #[unstable(feature = "collections",
               reason = "might get removed in favor of a more generic find in the future")]
    fn find_str(&self, needle: &str) -> Option<usize> {
        core_str::StrExt::find_str(&self[..], needle)
    }

    /// Retrieves the first character from a string slice and returns
    /// it. This does not allocate a new string; instead, it returns a
    /// slice that point one character beyond the character that was
    /// shifted. If the string does not contain any characters,
    /// None is returned instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// let (c, s1) = s.slice_shift_char().unwrap();
    /// assert_eq!(c, 'L');
    /// assert_eq!(s1, "öwe 老虎 Léopard");
    ///
    /// let (c, s2) = s1.slice_shift_char().unwrap();
    /// assert_eq!(c, 'ö');
    /// assert_eq!(s2, "we 老虎 Léopard");
    /// ```
    #[unstable(feature = "collections",
               reason = "awaiting conventions about shifting and slices")]
    fn slice_shift_char(&self) -> Option<(char, &str)> {
        core_str::StrExt::slice_shift_char(&self[..])
    }

    /// Returns the byte offset of an inner slice relative to an enclosing outer slice.
    ///
    /// Panics if `inner` is not a direct slice contained within self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let string = "a\nb\nc";
    /// let lines: Vec<&str> = string.lines().collect();
    ///
    /// assert!(string.subslice_offset(lines[0]) == 0); // &"a"
    /// assert!(string.subslice_offset(lines[1]) == 2); // &"b"
    /// assert!(string.subslice_offset(lines[2]) == 4); // &"c"
    /// ```
    #[unstable(feature = "collections",
               reason = "awaiting convention about comparability of arbitrary slices")]
    fn subslice_offset(&self, inner: &str) -> usize {
        core_str::StrExt::subslice_offset(&self[..], inner)
    }

    /// Return an unsafe pointer to the strings buffer.
    ///
    /// The caller must ensure that the string outlives this pointer,
    /// and that it is not reallocated (e.g. by pushing to the
    /// string).
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        core_str::StrExt::as_ptr(&self[..])
    }

    /// Return an iterator of `u16` over the string encoded as UTF-16.
    #[unstable(feature = "collections",
               reason = "this functionality may only be provided by libunicode")]
    fn utf16_units(&self) -> Utf16Units {
        Utf16Units { encoder: Utf16Encoder::new(self[..].chars()) }
    }

    /// Return the number of bytes in this string
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!("foo".len(), 3);
    /// assert_eq!("ƒoo".len(), 4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    fn len(&self) -> usize {
        core_str::StrExt::len(&self[..])
    }

    /// Returns true if this slice contains no bytes
    ///
    /// # Example
    ///
    /// ```
    /// assert!("".is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn is_empty(&self) -> bool {
        core_str::StrExt::is_empty(&self[..])
    }

    /// Parse this string into the specified type.
    ///
    /// # Example
    ///
    /// ```
    /// assert_eq!("4".parse::<u32>(), Ok(4));
    /// assert!("j".parse::<u32>().is_err());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
        core_str::StrExt::parse(&self[..])
    }

    /// Returns an iterator over the
    /// [grapheme clusters](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// of the string.
    ///
    /// If `is_extended` is true, the iterator is over the *extended grapheme clusters*;
    /// otherwise, the iterator is over the *legacy grapheme clusters*.
    /// [UAX#29](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// recommends extended grapheme cluster boundaries for general processing.
    ///
    /// # Example
    ///
    /// ```rust
    /// let gr1 = "a\u{310}e\u{301}o\u{308}\u{332}".graphemes(true).collect::<Vec<&str>>();
    /// let b: &[_] = &["a\u{310}", "e\u{301}", "o\u{308}\u{332}"];
    /// assert_eq!(gr1.as_slice(), b);
    /// let gr2 = "a\r\nb🇷🇺🇸🇹".graphemes(true).collect::<Vec<&str>>();
    /// let b: &[_] = &["a", "\r\n", "b", "🇷🇺🇸🇹"];
    /// assert_eq!(gr2.as_slice(), b);
    /// ```
    #[unstable(feature = "collections",
               reason = "this functionality may only be provided by libunicode")]
    fn graphemes(&self, is_extended: bool) -> Graphemes {
        UnicodeStr::graphemes(&self[..], is_extended)
    }

    /// Returns an iterator over the grapheme clusters of self and their byte offsets.
    /// See `graphemes()` method for more information.
    ///
    /// # Example
    ///
    /// ```rust
    /// let gr_inds = "a̐éö̲\r\n".grapheme_indices(true).collect::<Vec<(usize, &str)>>();
    /// let b: &[_] = &[(0, "a̐"), (3, "é"), (6, "ö̲"), (11, "\r\n")];
    /// assert_eq!(gr_inds.as_slice(), b);
    /// ```
    #[unstable(feature = "collections",
               reason = "this functionality may only be provided by libunicode")]
    fn grapheme_indices(&self, is_extended: bool) -> GraphemeIndices {
        UnicodeStr::grapheme_indices(&self[..], is_extended)
    }

    /// An iterator over the words of a string (subsequences separated
    /// by any sequence of whitespace). Sequences of whitespace are
    /// collapsed, so empty "words" are not included.
    ///
    /// # Example
    ///
    /// ```rust
    /// let some_words = " Mary   had\ta little  \n\t lamb";
    /// let v: Vec<&str> = some_words.words().collect();
    /// assert_eq!(v, vec!["Mary", "had", "a", "little", "lamb"]);
    /// ```
    #[unstable(feature = "str_words",
               reason = "the precise algorithm to use is unclear")]
    fn words(&self) -> Words {
        UnicodeStr::words(&self[..])
    }

    /// Returns a string's displayed width in columns, treating control
    /// characters as zero-width.
    ///
    /// `is_cjk` determines behavior for characters in the Ambiguous category:
    /// if `is_cjk` is `true`, these are 2 columns wide; otherwise, they are 1.
    /// In CJK locales, `is_cjk` should be `true`, else it should be `false`.
    /// [Unicode Standard Annex #11](http://www.unicode.org/reports/tr11/)
    /// recommends that these characters be treated as 1 column (i.e.,
    /// `is_cjk` = `false`) if the locale is unknown.
    #[unstable(feature = "collections",
               reason = "this functionality may only be provided by libunicode")]
    fn width(&self, is_cjk: bool) -> usize {
        UnicodeStr::width(&self[..], is_cjk)
    }

    /// Returns a string with leading and trailing whitespace removed.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trim(&self) -> &str {
        UnicodeStr::trim(&self[..])
    }

    /// Returns a string with leading whitespace removed.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trim_left(&self) -> &str {
        UnicodeStr::trim_left(&self[..])
    }

    /// Returns a string with trailing whitespace removed.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn trim_right(&self) -> &str {
        UnicodeStr::trim_right(&self[..])
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl StrExt for str {
    fn slice(&self, begin: usize, end: usize) -> &str {
        &self[begin..end]
    }

    fn slice_from(&self, begin: usize) -> &str {
        &self[begin..]
    }

    fn slice_to(&self, end: usize) -> &str {
        &self[..end]
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    use core::iter::AdditiveIterator;
    use super::from_utf8;
    use super::Utf8Error;

    #[test]
    fn test_le() {
        assert!("" <= "");
        assert!("" <= "foo");
        assert!("foo" <= "foo");
        assert!("foo" != "bar");
    }

    #[test]
    fn test_len() {
        assert_eq!("".len(), 0);
        assert_eq!("hello world".len(), 11);
        assert_eq!("\x63".len(), 1);
        assert_eq!("\u{a2}".len(), 2);
        assert_eq!("\u{3c0}".len(), 2);
        assert_eq!("\u{2620}".len(), 3);
        assert_eq!("\u{1d11e}".len(), 4);

        assert_eq!("".chars().count(), 0);
        assert_eq!("hello world".chars().count(), 11);
        assert_eq!("\x63".chars().count(), 1);
        assert_eq!("\u{a2}".chars().count(), 1);
        assert_eq!("\u{3c0}".chars().count(), 1);
        assert_eq!("\u{2620}".chars().count(), 1);
        assert_eq!("\u{1d11e}".chars().count(), 1);
        assert_eq!("ประเทศไทย中华Việt Nam".chars().count(), 19);

        assert_eq!("ｈｅｌｌｏ".width(false), 10);
        assert_eq!("ｈｅｌｌｏ".width(true), 10);
        assert_eq!("\0\0\0\0\0".width(false), 0);
        assert_eq!("\0\0\0\0\0".width(true), 0);
        assert_eq!("".width(false), 0);
        assert_eq!("".width(true), 0);
        assert_eq!("\u{2081}\u{2082}\u{2083}\u{2084}".width(false), 4);
        assert_eq!("\u{2081}\u{2082}\u{2083}\u{2084}".width(true), 8);
    }

    #[test]
    fn test_find() {
        assert_eq!("hello".find('l'), Some(2));
        assert_eq!("hello".find(|c:char| c == 'o'), Some(4));
        assert!("hello".find('x').is_none());
        assert!("hello".find(|c:char| c == 'x').is_none());
        assert_eq!("ประเทศไทย中华Việt Nam".find('华'), Some(30));
        assert_eq!("ประเทศไทย中华Việt Nam".find(|c: char| c == '华'), Some(30));
    }

    #[test]
    fn test_rfind() {
        assert_eq!("hello".rfind('l'), Some(3));
        assert_eq!("hello".rfind(|c:char| c == 'o'), Some(4));
        assert!("hello".rfind('x').is_none());
        assert!("hello".rfind(|c:char| c == 'x').is_none());
        assert_eq!("ประเทศไทย中华Việt Nam".rfind('华'), Some(30));
        assert_eq!("ประเทศไทย中华Việt Nam".rfind(|c: char| c == '华'), Some(30));
    }

    #[test]
    fn test_collect() {
        let empty = String::from_str("");
        let s: String = empty.chars().collect();
        assert_eq!(empty, s);
        let data = String::from_str("ประเทศไทย中");
        let s: String = data.chars().collect();
        assert_eq!(data, s);
    }

    #[test]
    fn test_into_bytes() {
        let data = String::from_str("asdf");
        let buf = data.into_bytes();
        assert_eq!(b"asdf", buf);
    }

    #[test]
    fn test_find_str() {
        // byte positions
        assert_eq!("".find_str(""), Some(0));
        assert!("banana".find_str("apple pie").is_none());

        let data = "abcabc";
        assert_eq!(data[0..6].find_str("ab"), Some(0));
        assert_eq!(data[2..6].find_str("ab"), Some(3 - 2));
        assert!(data[2..4].find_str("ab").is_none());

        let string = "ประเทศไทย中华Việt Nam";
        let mut data = String::from_str(string);
        data.push_str(string);
        assert!(data.find_str("ไท华").is_none());
        assert_eq!(data[0..43].find_str(""), Some(0));
        assert_eq!(data[6..43].find_str(""), Some(6 - 6));

        assert_eq!(data[0..43].find_str("ประ"), Some( 0));
        assert_eq!(data[0..43].find_str("ทศไ"), Some(12));
        assert_eq!(data[0..43].find_str("ย中"), Some(24));
        assert_eq!(data[0..43].find_str("iệt"), Some(34));
        assert_eq!(data[0..43].find_str("Nam"), Some(40));

        assert_eq!(data[43..86].find_str("ประ"), Some(43 - 43));
        assert_eq!(data[43..86].find_str("ทศไ"), Some(55 - 43));
        assert_eq!(data[43..86].find_str("ย中"), Some(67 - 43));
        assert_eq!(data[43..86].find_str("iệt"), Some(77 - 43));
        assert_eq!(data[43..86].find_str("Nam"), Some(83 - 43));
    }

    #[test]
    fn test_slice_chars() {
        fn t(a: &str, b: &str, start: usize) {
            assert_eq!(a.slice_chars(start, start + b.chars().count()), b);
        }
        t("", "", 0);
        t("hello", "llo", 2);
        t("hello", "el", 1);
        t("αβλ", "β", 1);
        t("αβλ", "", 3);
        assert_eq!("ะเทศไท", "ประเทศไทย中华Việt Nam".slice_chars(2, 8));
    }

    fn s(x: &str) -> String { x.to_string() }

    macro_rules! test_concat {
        ($expected: expr, $string: expr) => {
            {
                let s: String = $string.concat();
                assert_eq!($expected, s);
            }
        }
    }

    #[test]
    fn test_concat_for_different_types() {
        test_concat!("ab", vec![s("a"), s("b")]);
        test_concat!("ab", vec!["a", "b"]);
        test_concat!("ab", vec!["a", "b"]);
        test_concat!("ab", vec![s("a"), s("b")]);
    }

    #[test]
    fn test_concat_for_different_lengths() {
        let empty: &[&str] = &[];
        test_concat!("", empty);
        test_concat!("a", ["a"]);
        test_concat!("ab", ["a", "b"]);
        test_concat!("abc", ["", "a", "bc"]);
    }

    macro_rules! test_connect {
        ($expected: expr, $string: expr, $delim: expr) => {
            {
                let s = $string.connect($delim);
                assert_eq!($expected, s);
            }
        }
    }

    #[test]
    fn test_connect_for_different_types() {
        test_connect!("a-b", ["a", "b"], "-");
        let hyphen = "-".to_string();
        test_connect!("a-b", [s("a"), s("b")], &*hyphen);
        test_connect!("a-b", vec!["a", "b"], &*hyphen);
        test_connect!("a-b", &*vec!["a", "b"], "-");
        test_connect!("a-b", vec![s("a"), s("b")], "-");
    }

    #[test]
    fn test_connect_for_different_lengths() {
        let empty: &[&str] = &[];
        test_connect!("", empty, "-");
        test_connect!("a", ["a"], "-");
        test_connect!("a-b", ["a", "b"], "-");
        test_connect!("-a-bc", ["", "a", "bc"], "-");
    }

    #[test]
    fn test_unsafe_slice() {
        assert_eq!("ab", unsafe {"abc".slice_unchecked(0, 2)});
        assert_eq!("bc", unsafe {"abc".slice_unchecked(1, 3)});
        assert_eq!("", unsafe {"abc".slice_unchecked(1, 1)});
        fn a_million_letter_a() -> String {
            let mut i = 0;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("aaaaaaaaaa");
                i += 1;
            }
            rs
        }
        fn half_a_million_letter_a() -> String {
            let mut i = 0;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("aaaaa");
                i += 1;
            }
            rs
        }
        let letters = a_million_letter_a();
        assert!(half_a_million_letter_a() ==
            unsafe {String::from_str(letters.slice_unchecked(
                                     0,
                                     500000))});
    }

    #[test]
    fn test_starts_with() {
        assert!(("".starts_with("")));
        assert!(("abc".starts_with("")));
        assert!(("abc".starts_with("a")));
        assert!((!"a".starts_with("abc")));
        assert!((!"".starts_with("abc")));
        assert!((!"ödd".starts_with("-")));
        assert!(("ödd".starts_with("öd")));
    }

    #[test]
    fn test_ends_with() {
        assert!(("".ends_with("")));
        assert!(("abc".ends_with("")));
        assert!(("abc".ends_with("c")));
        assert!((!"a".ends_with("abc")));
        assert!((!"".ends_with("abc")));
        assert!((!"ddö".ends_with("-")));
        assert!(("ddö".ends_with("dö")));
    }

    #[test]
    fn test_is_empty() {
        assert!("".is_empty());
        assert!(!"a".is_empty());
    }

    #[test]
    fn test_replace() {
        let a = "a";
        assert_eq!("".replace(a, "b"), String::from_str(""));
        assert_eq!("a".replace(a, "b"), String::from_str("b"));
        assert_eq!("ab".replace(a, "b"), String::from_str("bb"));
        let test = "test";
        assert!(" test test ".replace(test, "toast") ==
            String::from_str(" toast toast "));
        assert_eq!(" test test ".replace(test, ""), String::from_str("   "));
    }

    #[test]
    fn test_replace_2a() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let a = "ประเ";
        let a2 = "دولة الكويتทศไทย中华";
        assert_eq!(data.replace(a, repl), a2);
    }

    #[test]
    fn test_replace_2b() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let b = "ะเ";
        let b2 = "ปรدولة الكويتทศไทย中华";
        assert_eq!(data.replace(b, repl), b2);
    }

    #[test]
    fn test_replace_2c() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let c = "中华";
        let c2 = "ประเทศไทยدولة الكويت";
        assert_eq!(data.replace(c, repl), c2);
    }

    #[test]
    fn test_replace_2d() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let d = "ไท华";
        assert_eq!(data.replace(d, repl), data);
    }

    #[test]
    fn test_slice() {
        assert_eq!("ab", "abc".slice(0, 2));
        assert_eq!("bc", "abc".slice(1, 3));
        assert_eq!("", "abc".slice(1, 1));
        assert_eq!("\u{65e5}", "\u{65e5}\u{672c}".slice(0, 3));

        let data = "ประเทศไทย中华";
        assert_eq!("ป", data.slice(0, 3));
        assert_eq!("ร", data.slice(3, 6));
        assert_eq!("", data.slice(3, 3));
        assert_eq!("华", data.slice(30, 33));

        fn a_million_letter_x() -> String {
            let mut i = 0;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("华华华华华华华华华华");
                i += 1;
            }
            rs
        }
        fn half_a_million_letter_x() -> String {
            let mut i = 0;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("华华华华华");
                i += 1;
            }
            rs
        }
        let letters = a_million_letter_x();
        assert!(half_a_million_letter_x() ==
            String::from_str(letters.slice(0, 3 * 500000)));
    }

    #[test]
    fn test_slice_2() {
        let ss = "中华Việt Nam";

        assert_eq!("华", ss.slice(3, 6));
        assert_eq!("Việt Nam", ss.slice(6, 16));

        assert_eq!("ab", "abc".slice(0, 2));
        assert_eq!("bc", "abc".slice(1, 3));
        assert_eq!("", "abc".slice(1, 1));

        assert_eq!("中", ss.slice(0, 3));
        assert_eq!("华V", ss.slice(3, 7));
        assert_eq!("", ss.slice(3, 3));
        /*0: 中
          3: 华
          6: V
          7: i
          8: ệ
         11: t
         12:
         13: N
         14: a
         15: m */
    }

    #[test]
    #[should_fail]
    fn test_slice_fail() {
        "中华Việt Nam".slice(0, 2);
    }

    #[test]
    fn test_slice_from() {
        assert_eq!("abcd".slice_from(0), "abcd");
        assert_eq!("abcd".slice_from(2), "cd");
        assert_eq!("abcd".slice_from(4), "");
    }
    #[test]
    fn test_slice_to() {
        assert_eq!("abcd".slice_to(0), "");
        assert_eq!("abcd".slice_to(2), "ab");
        assert_eq!("abcd".slice_to(4), "abcd");
    }

    #[test]
    fn test_trim_left_matches() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_left_matches(v), " *** foo *** ");
        let chars: &[char] = &['*', ' '];
        assert_eq!(" *** foo *** ".trim_left_matches(chars), "foo *** ");
        assert_eq!(" ***  *** ".trim_left_matches(chars), "");
        assert_eq!("foo *** ".trim_left_matches(chars), "foo *** ");

        assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
        let chars: &[char] = &['1', '2'];
        assert_eq!("12foo1bar12".trim_left_matches(chars), "foo1bar12");
        assert_eq!("123foo1bar123".trim_left_matches(|c: char| c.is_numeric()), "foo1bar123");
    }

    #[test]
    fn test_trim_right_matches() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_right_matches(v), " *** foo *** ");
        let chars: &[char] = &['*', ' '];
        assert_eq!(" *** foo *** ".trim_right_matches(chars), " *** foo");
        assert_eq!(" ***  *** ".trim_right_matches(chars), "");
        assert_eq!(" *** foo".trim_right_matches(chars), " *** foo");

        assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
        let chars: &[char] = &['1', '2'];
        assert_eq!("12foo1bar12".trim_right_matches(chars), "12foo1bar");
        assert_eq!("123foo1bar123".trim_right_matches(|c: char| c.is_numeric()), "123foo1bar");
    }

    #[test]
    fn test_trim_matches() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_matches(v), " *** foo *** ");
        let chars: &[char] = &['*', ' '];
        assert_eq!(" *** foo *** ".trim_matches(chars), "foo");
        assert_eq!(" ***  *** ".trim_matches(chars), "");
        assert_eq!("foo".trim_matches(chars), "foo");

        assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
        let chars: &[char] = &['1', '2'];
        assert_eq!("12foo1bar12".trim_matches(chars), "foo1bar");
        assert_eq!("123foo1bar123".trim_matches(|c: char| c.is_numeric()), "foo1bar");
    }

    #[test]
    fn test_trim_left() {
        assert_eq!("".trim_left(), "");
        assert_eq!("a".trim_left(), "a");
        assert_eq!("    ".trim_left(), "");
        assert_eq!("     blah".trim_left(), "blah");
        assert_eq!("   \u{3000}  wut".trim_left(), "wut");
        assert_eq!("hey ".trim_left(), "hey ");
    }

    #[test]
    fn test_trim_right() {
        assert_eq!("".trim_right(), "");
        assert_eq!("a".trim_right(), "a");
        assert_eq!("    ".trim_right(), "");
        assert_eq!("blah     ".trim_right(), "blah");
        assert_eq!("wut   \u{3000}  ".trim_right(), "wut");
        assert_eq!(" hey".trim_right(), " hey");
    }

    #[test]
    fn test_trim() {
        assert_eq!("".trim(), "");
        assert_eq!("a".trim(), "a");
        assert_eq!("    ".trim(), "");
        assert_eq!("    blah     ".trim(), "blah");
        assert_eq!("\nwut   \u{3000}  ".trim(), "wut");
        assert_eq!(" hey dude ".trim(), "hey dude");
    }

    #[test]
    fn test_is_whitespace() {
        assert!("".chars().all(|c| c.is_whitespace()));
        assert!(" ".chars().all(|c| c.is_whitespace()));
        assert!("\u{2009}".chars().all(|c| c.is_whitespace())); // Thin space
        assert!("  \n\t   ".chars().all(|c| c.is_whitespace()));
        assert!(!"   _   ".chars().all(|c| c.is_whitespace()));
    }

    #[test]
    fn test_slice_shift_char() {
        let data = "ประเทศไทย中";
        assert_eq!(data.slice_shift_char(), Some(('ป', "ระเทศไทย中")));
    }

    #[test]
    fn test_slice_shift_char_2() {
        let empty = "";
        assert_eq!(empty.slice_shift_char(), None);
    }

    #[test]
    fn test_is_utf8() {
        // deny overlong encodings
        assert!(from_utf8(&[0xc0, 0x80]).is_err());
        assert!(from_utf8(&[0xc0, 0xae]).is_err());
        assert!(from_utf8(&[0xe0, 0x80, 0x80]).is_err());
        assert!(from_utf8(&[0xe0, 0x80, 0xaf]).is_err());
        assert!(from_utf8(&[0xe0, 0x81, 0x81]).is_err());
        assert!(from_utf8(&[0xf0, 0x82, 0x82, 0xac]).is_err());
        assert!(from_utf8(&[0xf4, 0x90, 0x80, 0x80]).is_err());

        // deny surrogates
        assert!(from_utf8(&[0xED, 0xA0, 0x80]).is_err());
        assert!(from_utf8(&[0xED, 0xBF, 0xBF]).is_err());

        assert!(from_utf8(&[0xC2, 0x80]).is_ok());
        assert!(from_utf8(&[0xDF, 0xBF]).is_ok());
        assert!(from_utf8(&[0xE0, 0xA0, 0x80]).is_ok());
        assert!(from_utf8(&[0xED, 0x9F, 0xBF]).is_ok());
        assert!(from_utf8(&[0xEE, 0x80, 0x80]).is_ok());
        assert!(from_utf8(&[0xEF, 0xBF, 0xBF]).is_ok());
        assert!(from_utf8(&[0xF0, 0x90, 0x80, 0x80]).is_ok());
        assert!(from_utf8(&[0xF4, 0x8F, 0xBF, 0xBF]).is_ok());
    }

    #[test]
    fn test_is_utf16() {
        use unicode::str::is_utf16;
        macro_rules! pos {
            ($($e:expr),*) => { { $(assert!(is_utf16($e));)* } }
        }

        // non-surrogates
        pos!(&[0x0000],
             &[0x0001, 0x0002],
             &[0xD7FF],
             &[0xE000]);

        // surrogate pairs (randomly generated with Python 3's
        // .encode('utf-16be'))
        pos!(&[0xdb54, 0xdf16, 0xd880, 0xdee0, 0xdb6a, 0xdd45],
             &[0xd91f, 0xdeb1, 0xdb31, 0xdd84, 0xd8e2, 0xde14],
             &[0xdb9f, 0xdc26, 0xdb6f, 0xde58, 0xd850, 0xdfae]);

        // mixtures (also random)
        pos!(&[0xd921, 0xdcc2, 0x002d, 0x004d, 0xdb32, 0xdf65],
             &[0xdb45, 0xdd2d, 0x006a, 0xdacd, 0xddfe, 0x0006],
             &[0x0067, 0xd8ff, 0xddb7, 0x000f, 0xd900, 0xdc80]);

        // negative tests
        macro_rules! neg {
            ($($e:expr),*) => { { $(assert!(!is_utf16($e));)* } }
        }

        neg!(
            // surrogate + regular unit
            &[0xdb45, 0x0000],
            // surrogate + lead surrogate
            &[0xd900, 0xd900],
            // unterminated surrogate
            &[0xd8ff],
            // trail surrogate without a lead
            &[0xddb7]);

        // random byte sequences that Python 3's .decode('utf-16be')
        // failed on
        neg!(&[0x5b3d, 0x0141, 0xde9e, 0x8fdc, 0xc6e7],
             &[0xdf5a, 0x82a5, 0x62b9, 0xb447, 0x92f3],
             &[0xda4e, 0x42bc, 0x4462, 0xee98, 0xc2ca],
             &[0xbe00, 0xb04a, 0x6ecb, 0xdd89, 0xe278],
             &[0x0465, 0xab56, 0xdbb6, 0xa893, 0x665e],
             &[0x6b7f, 0x0a19, 0x40f4, 0xa657, 0xdcc5],
             &[0x9b50, 0xda5e, 0x24ec, 0x03ad, 0x6dee],
             &[0x8d17, 0xcaa7, 0xf4ae, 0xdf6e, 0xbed7],
             &[0xdaee, 0x2584, 0x7d30, 0xa626, 0x121a],
             &[0xd956, 0x4b43, 0x7570, 0xccd6, 0x4f4a],
             &[0x9dcf, 0x1b49, 0x4ba5, 0xfce9, 0xdffe],
             &[0x6572, 0xce53, 0xb05a, 0xf6af, 0xdacf],
             &[0x1b90, 0x728c, 0x9906, 0xdb68, 0xf46e],
             &[0x1606, 0xbeca, 0xbe76, 0x860f, 0xdfa5],
             &[0x8b4f, 0xde7a, 0xd220, 0x9fac, 0x2b6f],
             &[0xb8fe, 0xebbe, 0xda32, 0x1a5f, 0x8b8b],
             &[0x934b, 0x8956, 0xc434, 0x1881, 0xddf7],
             &[0x5a95, 0x13fc, 0xf116, 0xd89b, 0x93f9],
             &[0xd640, 0x71f1, 0xdd7d, 0x77eb, 0x1cd8],
             &[0x348b, 0xaef0, 0xdb2c, 0xebf1, 0x1282],
             &[0x50d7, 0xd824, 0x5010, 0xb369, 0x22ea]);
    }

    #[test]
    fn test_as_bytes() {
        // no null
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        let b: &[u8] = &[];
        assert_eq!("".as_bytes(), b);
        assert_eq!("abc".as_bytes(), b"abc");
        assert_eq!("ศไทย中华Việt Nam".as_bytes(), v);
    }

    #[test]
    #[should_fail]
    fn test_as_bytes_fail() {
        // Don't double free. (I'm not sure if this exercises the
        // original problem code path anymore.)
        let s = String::from_str("");
        let _bytes = s.as_bytes();
        panic!();
    }

    #[test]
    fn test_as_ptr() {
        let buf = "hello".as_ptr();
        unsafe {
            assert_eq!(*buf.offset(0), b'h');
            assert_eq!(*buf.offset(1), b'e');
            assert_eq!(*buf.offset(2), b'l');
            assert_eq!(*buf.offset(3), b'l');
            assert_eq!(*buf.offset(4), b'o');
        }
    }

    #[test]
    fn test_subslice_offset() {
        let a = "kernelsprite";
        let b = &a[7..a.len()];
        let c = &a[0..a.len() - 6];
        assert_eq!(a.subslice_offset(b), 7);
        assert_eq!(a.subslice_offset(c), 0);

        let string = "a\nb\nc";
        let lines: Vec<&str> = string.lines().collect();
        assert_eq!(string.subslice_offset(lines[0]), 0);
        assert_eq!(string.subslice_offset(lines[1]), 2);
        assert_eq!(string.subslice_offset(lines[2]), 4);
    }

    #[test]
    #[should_fail]
    fn test_subslice_offset_2() {
        let a = "alchemiter";
        let b = "cruxtruder";
        a.subslice_offset(b);
    }

    #[test]
    fn vec_str_conversions() {
        let s1: String = String::from_str("All mimsy were the borogoves");

        let v: Vec<u8> = s1.as_bytes().to_vec();
        let s2: String = String::from_str(from_utf8(&v).unwrap());
        let mut i = 0;
        let n1 = s1.len();
        let n2 = v.len();
        assert_eq!(n1, n2);
        while i < n1 {
            let a: u8 = s1.as_bytes()[i];
            let b: u8 = s2.as_bytes()[i];
            debug!("{}", a);
            debug!("{}", b);
            assert_eq!(a, b);
            i += 1;
        }
    }

    #[test]
    fn test_contains() {
        assert!("abcde".contains("bcd"));
        assert!("abcde".contains("abcd"));
        assert!("abcde".contains("bcde"));
        assert!("abcde".contains(""));
        assert!("".contains(""));
        assert!(!"abcde".contains("def"));
        assert!(!"".contains("a"));

        let data = "ประเทศไทย中华Việt Nam";
        assert!(data.contains("ประเ"));
        assert!(data.contains("ะเ"));
        assert!(data.contains("中华"));
        assert!(!data.contains("ไท华"));
    }

    #[test]
    fn test_contains_char() {
        assert!("abc".contains_char('b'));
        assert!("a".contains_char('a'));
        assert!(!"abc".contains_char('d'));
        assert!(!"".contains_char('a'));
    }

    #[test]
    fn test_char_at() {
        let s = "ศไทย中华Việt Nam";
        let v = vec!['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
        let mut pos = 0;
        for ch in &v {
            assert!(s.char_at(pos) == *ch);
            pos += ch.to_string().len();
        }
    }

    #[test]
    fn test_char_at_reverse() {
        let s = "ศไทย中华Việt Nam";
        let v = vec!['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
        let mut pos = s.len();
        for ch in v.iter().rev() {
            assert!(s.char_at_reverse(pos) == *ch);
            pos -= ch.to_string().len();
        }
    }

    #[test]
    fn test_escape_unicode() {
        assert_eq!("abc".escape_unicode(),
                   String::from_str("\\u{61}\\u{62}\\u{63}"));
        assert_eq!("a c".escape_unicode(),
                   String::from_str("\\u{61}\\u{20}\\u{63}"));
        assert_eq!("\r\n\t".escape_unicode(),
                   String::from_str("\\u{d}\\u{a}\\u{9}"));
        assert_eq!("'\"\\".escape_unicode(),
                   String::from_str("\\u{27}\\u{22}\\u{5c}"));
        assert_eq!("\x00\x01\u{fe}\u{ff}".escape_unicode(),
                   String::from_str("\\u{0}\\u{1}\\u{fe}\\u{ff}"));
        assert_eq!("\u{100}\u{ffff}".escape_unicode(),
                   String::from_str("\\u{100}\\u{ffff}"));
        assert_eq!("\u{10000}\u{10ffff}".escape_unicode(),
                   String::from_str("\\u{10000}\\u{10ffff}"));
        assert_eq!("ab\u{fb00}".escape_unicode(),
                   String::from_str("\\u{61}\\u{62}\\u{fb00}"));
        assert_eq!("\u{1d4ea}\r".escape_unicode(),
                   String::from_str("\\u{1d4ea}\\u{d}"));
    }

    #[test]
    fn test_escape_default() {
        assert_eq!("abc".escape_default(), String::from_str("abc"));
        assert_eq!("a c".escape_default(), String::from_str("a c"));
        assert_eq!("\r\n\t".escape_default(), String::from_str("\\r\\n\\t"));
        assert_eq!("'\"\\".escape_default(), String::from_str("\\'\\\"\\\\"));
        assert_eq!("\u{100}\u{ffff}".escape_default(),
                   String::from_str("\\u{100}\\u{ffff}"));
        assert_eq!("\u{10000}\u{10ffff}".escape_default(),
                   String::from_str("\\u{10000}\\u{10ffff}"));
        assert_eq!("ab\u{fb00}".escape_default(),
                   String::from_str("ab\\u{fb00}"));
        assert_eq!("\u{1d4ea}\r".escape_default(),
                   String::from_str("\\u{1d4ea}\\r"));
    }

    #[test]
    fn test_total_ord() {
        "1234".cmp("123") == Greater;
        "123".cmp("1234") == Less;
        "1234".cmp("1234") == Equal;
        "12345555".cmp("123456") == Less;
        "22".cmp("1234") == Greater;
    }

    #[test]
    fn test_char_range_at() {
        let data = "b¢€𤭢𤭢€¢b";
        assert_eq!('b', data.char_range_at(0).ch);
        assert_eq!('¢', data.char_range_at(1).ch);
        assert_eq!('€', data.char_range_at(3).ch);
        assert_eq!('𤭢', data.char_range_at(6).ch);
        assert_eq!('𤭢', data.char_range_at(10).ch);
        assert_eq!('€', data.char_range_at(14).ch);
        assert_eq!('¢', data.char_range_at(17).ch);
        assert_eq!('b', data.char_range_at(19).ch);
    }

    #[test]
    fn test_char_range_at_reverse_underflow() {
        assert_eq!("abc".char_range_at_reverse(0).next, 0);
    }

    #[test]
    fn test_iterator() {
        let s = "ศไทย中华Việt Nam";
        let v = ['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

        let mut pos = 0;
        let it = s.chars();

        for c in it {
            assert_eq!(c, v[pos]);
            pos += 1;
        }
        assert_eq!(pos, v.len());
    }

    #[test]
    fn test_rev_iterator() {
        let s = "ศไทย中华Việt Nam";
        let v = ['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

        let mut pos = 0;
        let it = s.chars().rev();

        for c in it {
            assert_eq!(c, v[pos]);
            pos += 1;
        }
        assert_eq!(pos, v.len());
    }

    #[test]
    fn test_chars_decoding() {
        let mut bytes = [0u8; 4];
        for c in (0u32..0x110000).filter_map(|c| ::core::char::from_u32(c)) {
            let len = c.encode_utf8(&mut bytes).unwrap_or(0);
            let s = ::core::str::from_utf8(&bytes[..len]).unwrap();
            if Some(c) != s.chars().next() {
                panic!("character {:x}={} does not decode correctly", c as u32, c);
            }
        }
    }

    #[test]
    fn test_chars_rev_decoding() {
        let mut bytes = [0u8; 4];
        for c in (0u32..0x110000).filter_map(|c| ::core::char::from_u32(c)) {
            let len = c.encode_utf8(&mut bytes).unwrap_or(0);
            let s = ::core::str::from_utf8(&bytes[..len]).unwrap();
            if Some(c) != s.chars().rev().next() {
                panic!("character {:x}={} does not decode correctly", c as u32, c);
            }
        }
    }

    #[test]
    fn test_iterator_clone() {
        let s = "ศไทย中华Việt Nam";
        let mut it = s.chars();
        it.next();
        assert!(it.clone().zip(it).all(|(x,y)| x == y));
    }

    #[test]
    fn test_bytesator() {
        let s = "ศไทย中华Việt Nam";
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        let mut pos = 0;

        for b in s.bytes() {
            assert_eq!(b, v[pos]);
            pos += 1;
        }
    }

    #[test]
    fn test_bytes_revator() {
        let s = "ศไทย中华Việt Nam";
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        let mut pos = v.len();

        for b in s.bytes().rev() {
            pos -= 1;
            assert_eq!(b, v[pos]);
        }
    }

    #[test]
    fn test_char_indicesator() {
        let s = "ศไทย中华Việt Nam";
        let p = [0, 3, 6, 9, 12, 15, 18, 19, 20, 23, 24, 25, 26, 27];
        let v = ['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

        let mut pos = 0;
        let it = s.char_indices();

        for c in it {
            assert_eq!(c, (p[pos], v[pos]));
            pos += 1;
        }
        assert_eq!(pos, v.len());
        assert_eq!(pos, p.len());
    }

    #[test]
    fn test_char_indices_revator() {
        let s = "ศไทย中华Việt Nam";
        let p = [27, 26, 25, 24, 23, 20, 19, 18, 15, 12, 9, 6, 3, 0];
        let v = ['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

        let mut pos = 0;
        let it = s.char_indices().rev();

        for c in it {
            assert_eq!(c, (p[pos], v[pos]));
            pos += 1;
        }
        assert_eq!(pos, v.len());
        assert_eq!(pos, p.len());
    }

    #[test]
    fn test_splitn_char_iterator() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: Vec<&str> = data.splitn(3, ' ').collect();
        assert_eq!(split, vec!["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

        let split: Vec<&str> = data.splitn(3, |c: char| c == ' ').collect();
        assert_eq!(split, vec!["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

        // Unicode
        let split: Vec<&str> = data.splitn(3, 'ä').collect();
        assert_eq!(split, vec!["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);

        let split: Vec<&str> = data.splitn(3, |c: char| c == 'ä').collect();
        assert_eq!(split, vec!["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);
    }

    #[test]
    fn test_split_char_iterator_no_trailing() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: Vec<&str> = data.split('\n').collect();
        assert_eq!(split, vec!["", "Märy häd ä little lämb", "Little lämb", ""]);

        let split: Vec<&str> = data.split_terminator('\n').collect();
        assert_eq!(split, vec!["", "Märy häd ä little lämb", "Little lämb"]);
    }

    #[test]
    fn test_words() {
        let data = "\n \tMäry   häd\tä  little lämb\nLittle lämb\n";
        let words: Vec<&str> = data.words().collect();
        assert_eq!(words, vec!["Märy", "häd", "ä", "little", "lämb", "Little", "lämb"])
    }

    #[test]
    fn test_nfd_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfd_chars().collect::<String>(), $expected);
            }
        }
        t!("abc", "abc");
        t!("\u{1e0b}\u{1c4}", "d\u{307}\u{1c4}");
        t!("\u{2026}", "\u{2026}");
        t!("\u{2126}", "\u{3a9}");
        t!("\u{1e0b}\u{323}", "d\u{323}\u{307}");
        t!("\u{1e0d}\u{307}", "d\u{323}\u{307}");
        t!("a\u{301}", "a\u{301}");
        t!("\u{301}a", "\u{301}a");
        t!("\u{d4db}", "\u{1111}\u{1171}\u{11b6}");
        t!("\u{ac1c}", "\u{1100}\u{1162}");
    }

    #[test]
    fn test_nfkd_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfkd_chars().collect::<String>(), $expected);
            }
        }
        t!("abc", "abc");
        t!("\u{1e0b}\u{1c4}", "d\u{307}DZ\u{30c}");
        t!("\u{2026}", "...");
        t!("\u{2126}", "\u{3a9}");
        t!("\u{1e0b}\u{323}", "d\u{323}\u{307}");
        t!("\u{1e0d}\u{307}", "d\u{323}\u{307}");
        t!("a\u{301}", "a\u{301}");
        t!("\u{301}a", "\u{301}a");
        t!("\u{d4db}", "\u{1111}\u{1171}\u{11b6}");
        t!("\u{ac1c}", "\u{1100}\u{1162}");
    }

    #[test]
    fn test_nfc_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfc_chars().collect::<String>(), $expected);
            }
        }
        t!("abc", "abc");
        t!("\u{1e0b}\u{1c4}", "\u{1e0b}\u{1c4}");
        t!("\u{2026}", "\u{2026}");
        t!("\u{2126}", "\u{3a9}");
        t!("\u{1e0b}\u{323}", "\u{1e0d}\u{307}");
        t!("\u{1e0d}\u{307}", "\u{1e0d}\u{307}");
        t!("a\u{301}", "\u{e1}");
        t!("\u{301}a", "\u{301}a");
        t!("\u{d4db}", "\u{d4db}");
        t!("\u{ac1c}", "\u{ac1c}");
        t!("a\u{300}\u{305}\u{315}\u{5ae}b", "\u{e0}\u{5ae}\u{305}\u{315}b");
    }

    #[test]
    fn test_nfkc_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfkc_chars().collect::<String>(), $expected);
            }
        }
        t!("abc", "abc");
        t!("\u{1e0b}\u{1c4}", "\u{1e0b}D\u{17d}");
        t!("\u{2026}", "...");
        t!("\u{2126}", "\u{3a9}");
        t!("\u{1e0b}\u{323}", "\u{1e0d}\u{307}");
        t!("\u{1e0d}\u{307}", "\u{1e0d}\u{307}");
        t!("a\u{301}", "\u{e1}");
        t!("\u{301}a", "\u{301}a");
        t!("\u{d4db}", "\u{d4db}");
        t!("\u{ac1c}", "\u{ac1c}");
        t!("a\u{300}\u{305}\u{315}\u{5ae}b", "\u{e0}\u{5ae}\u{305}\u{315}b");
    }

    #[test]
    fn test_lines() {
        let data = "\nMäry häd ä little lämb\n\nLittle lämb\n";
        let lines: Vec<&str> = data.lines().collect();
        assert_eq!(lines, vec!["", "Märy häd ä little lämb", "", "Little lämb"]);

        let data = "\nMäry häd ä little lämb\n\nLittle lämb"; // no trailing \n
        let lines: Vec<&str> = data.lines().collect();
        assert_eq!(lines, vec!["", "Märy häd ä little lämb", "", "Little lämb"]);
    }

    #[test]
    fn test_graphemes() {
        use core::iter::order;
        // official Unicode test data
        // from http://www.unicode.org/Public/UCD/latest/ucd/auxiliary/GraphemeBreakTest.txt
        let test_same: [(_, &[_]); 325] = [
            ("\u{20}\u{20}", &["\u{20}", "\u{20}"]),
            ("\u{20}\u{308}\u{20}", &["\u{20}\u{308}", "\u{20}"]),
            ("\u{20}\u{D}", &["\u{20}", "\u{D}"]),
            ("\u{20}\u{308}\u{D}", &["\u{20}\u{308}", "\u{D}"]),
            ("\u{20}\u{A}", &["\u{20}", "\u{A}"]),
            ("\u{20}\u{308}\u{A}", &["\u{20}\u{308}", "\u{A}"]),
            ("\u{20}\u{1}", &["\u{20}", "\u{1}"]),
            ("\u{20}\u{308}\u{1}", &["\u{20}\u{308}", "\u{1}"]),
            ("\u{20}\u{300}", &["\u{20}\u{300}"]),
            ("\u{20}\u{308}\u{300}", &["\u{20}\u{308}\u{300}"]),
            ("\u{20}\u{1100}", &["\u{20}", "\u{1100}"]),
            ("\u{20}\u{308}\u{1100}", &["\u{20}\u{308}", "\u{1100}"]),
            ("\u{20}\u{1160}", &["\u{20}", "\u{1160}"]),
            ("\u{20}\u{308}\u{1160}", &["\u{20}\u{308}", "\u{1160}"]),
            ("\u{20}\u{11A8}", &["\u{20}", "\u{11A8}"]),
            ("\u{20}\u{308}\u{11A8}", &["\u{20}\u{308}", "\u{11A8}"]),
            ("\u{20}\u{AC00}", &["\u{20}", "\u{AC00}"]),
            ("\u{20}\u{308}\u{AC00}", &["\u{20}\u{308}", "\u{AC00}"]),
            ("\u{20}\u{AC01}", &["\u{20}", "\u{AC01}"]),
            ("\u{20}\u{308}\u{AC01}", &["\u{20}\u{308}", "\u{AC01}"]),
            ("\u{20}\u{1F1E6}", &["\u{20}", "\u{1F1E6}"]),
            ("\u{20}\u{308}\u{1F1E6}", &["\u{20}\u{308}", "\u{1F1E6}"]),
            ("\u{20}\u{378}", &["\u{20}", "\u{378}"]),
            ("\u{20}\u{308}\u{378}", &["\u{20}\u{308}", "\u{378}"]),
            ("\u{D}\u{20}", &["\u{D}", "\u{20}"]),
            ("\u{D}\u{308}\u{20}", &["\u{D}", "\u{308}", "\u{20}"]),
            ("\u{D}\u{D}", &["\u{D}", "\u{D}"]),
            ("\u{D}\u{308}\u{D}", &["\u{D}", "\u{308}", "\u{D}"]),
            ("\u{D}\u{A}", &["\u{D}\u{A}"]),
            ("\u{D}\u{308}\u{A}", &["\u{D}", "\u{308}", "\u{A}"]),
            ("\u{D}\u{1}", &["\u{D}", "\u{1}"]),
            ("\u{D}\u{308}\u{1}", &["\u{D}", "\u{308}", "\u{1}"]),
            ("\u{D}\u{300}", &["\u{D}", "\u{300}"]),
            ("\u{D}\u{308}\u{300}", &["\u{D}", "\u{308}\u{300}"]),
            ("\u{D}\u{903}", &["\u{D}", "\u{903}"]),
            ("\u{D}\u{1100}", &["\u{D}", "\u{1100}"]),
            ("\u{D}\u{308}\u{1100}", &["\u{D}", "\u{308}", "\u{1100}"]),
            ("\u{D}\u{1160}", &["\u{D}", "\u{1160}"]),
            ("\u{D}\u{308}\u{1160}", &["\u{D}", "\u{308}", "\u{1160}"]),
            ("\u{D}\u{11A8}", &["\u{D}", "\u{11A8}"]),
            ("\u{D}\u{308}\u{11A8}", &["\u{D}", "\u{308}", "\u{11A8}"]),
            ("\u{D}\u{AC00}", &["\u{D}", "\u{AC00}"]),
            ("\u{D}\u{308}\u{AC00}", &["\u{D}", "\u{308}", "\u{AC00}"]),
            ("\u{D}\u{AC01}", &["\u{D}", "\u{AC01}"]),
            ("\u{D}\u{308}\u{AC01}", &["\u{D}", "\u{308}", "\u{AC01}"]),
            ("\u{D}\u{1F1E6}", &["\u{D}", "\u{1F1E6}"]),
            ("\u{D}\u{308}\u{1F1E6}", &["\u{D}", "\u{308}", "\u{1F1E6}"]),
            ("\u{D}\u{378}", &["\u{D}", "\u{378}"]),
            ("\u{D}\u{308}\u{378}", &["\u{D}", "\u{308}", "\u{378}"]),
            ("\u{A}\u{20}", &["\u{A}", "\u{20}"]),
            ("\u{A}\u{308}\u{20}", &["\u{A}", "\u{308}", "\u{20}"]),
            ("\u{A}\u{D}", &["\u{A}", "\u{D}"]),
            ("\u{A}\u{308}\u{D}", &["\u{A}", "\u{308}", "\u{D}"]),
            ("\u{A}\u{A}", &["\u{A}", "\u{A}"]),
            ("\u{A}\u{308}\u{A}", &["\u{A}", "\u{308}", "\u{A}"]),
            ("\u{A}\u{1}", &["\u{A}", "\u{1}"]),
            ("\u{A}\u{308}\u{1}", &["\u{A}", "\u{308}", "\u{1}"]),
            ("\u{A}\u{300}", &["\u{A}", "\u{300}"]),
            ("\u{A}\u{308}\u{300}", &["\u{A}", "\u{308}\u{300}"]),
            ("\u{A}\u{903}", &["\u{A}", "\u{903}"]),
            ("\u{A}\u{1100}", &["\u{A}", "\u{1100}"]),
            ("\u{A}\u{308}\u{1100}", &["\u{A}", "\u{308}", "\u{1100}"]),
            ("\u{A}\u{1160}", &["\u{A}", "\u{1160}"]),
            ("\u{A}\u{308}\u{1160}", &["\u{A}", "\u{308}", "\u{1160}"]),
            ("\u{A}\u{11A8}", &["\u{A}", "\u{11A8}"]),
            ("\u{A}\u{308}\u{11A8}", &["\u{A}", "\u{308}", "\u{11A8}"]),
            ("\u{A}\u{AC00}", &["\u{A}", "\u{AC00}"]),
            ("\u{A}\u{308}\u{AC00}", &["\u{A}", "\u{308}", "\u{AC00}"]),
            ("\u{A}\u{AC01}", &["\u{A}", "\u{AC01}"]),
            ("\u{A}\u{308}\u{AC01}", &["\u{A}", "\u{308}", "\u{AC01}"]),
            ("\u{A}\u{1F1E6}", &["\u{A}", "\u{1F1E6}"]),
            ("\u{A}\u{308}\u{1F1E6}", &["\u{A}", "\u{308}", "\u{1F1E6}"]),
            ("\u{A}\u{378}", &["\u{A}", "\u{378}"]),
            ("\u{A}\u{308}\u{378}", &["\u{A}", "\u{308}", "\u{378}"]),
            ("\u{1}\u{20}", &["\u{1}", "\u{20}"]),
            ("\u{1}\u{308}\u{20}", &["\u{1}", "\u{308}", "\u{20}"]),
            ("\u{1}\u{D}", &["\u{1}", "\u{D}"]),
            ("\u{1}\u{308}\u{D}", &["\u{1}", "\u{308}", "\u{D}"]),
            ("\u{1}\u{A}", &["\u{1}", "\u{A}"]),
            ("\u{1}\u{308}\u{A}", &["\u{1}", "\u{308}", "\u{A}"]),
            ("\u{1}\u{1}", &["\u{1}", "\u{1}"]),
            ("\u{1}\u{308}\u{1}", &["\u{1}", "\u{308}", "\u{1}"]),
            ("\u{1}\u{300}", &["\u{1}", "\u{300}"]),
            ("\u{1}\u{308}\u{300}", &["\u{1}", "\u{308}\u{300}"]),
            ("\u{1}\u{903}", &["\u{1}", "\u{903}"]),
            ("\u{1}\u{1100}", &["\u{1}", "\u{1100}"]),
            ("\u{1}\u{308}\u{1100}", &["\u{1}", "\u{308}", "\u{1100}"]),
            ("\u{1}\u{1160}", &["\u{1}", "\u{1160}"]),
            ("\u{1}\u{308}\u{1160}", &["\u{1}", "\u{308}", "\u{1160}"]),
            ("\u{1}\u{11A8}", &["\u{1}", "\u{11A8}"]),
            ("\u{1}\u{308}\u{11A8}", &["\u{1}", "\u{308}", "\u{11A8}"]),
            ("\u{1}\u{AC00}", &["\u{1}", "\u{AC00}"]),
            ("\u{1}\u{308}\u{AC00}", &["\u{1}", "\u{308}", "\u{AC00}"]),
            ("\u{1}\u{AC01}", &["\u{1}", "\u{AC01}"]),
            ("\u{1}\u{308}\u{AC01}", &["\u{1}", "\u{308}", "\u{AC01}"]),
            ("\u{1}\u{1F1E6}", &["\u{1}", "\u{1F1E6}"]),
            ("\u{1}\u{308}\u{1F1E6}", &["\u{1}", "\u{308}", "\u{1F1E6}"]),
            ("\u{1}\u{378}", &["\u{1}", "\u{378}"]),
            ("\u{1}\u{308}\u{378}", &["\u{1}", "\u{308}", "\u{378}"]),
            ("\u{300}\u{20}", &["\u{300}", "\u{20}"]),
            ("\u{300}\u{308}\u{20}", &["\u{300}\u{308}", "\u{20}"]),
            ("\u{300}\u{D}", &["\u{300}", "\u{D}"]),
            ("\u{300}\u{308}\u{D}", &["\u{300}\u{308}", "\u{D}"]),
            ("\u{300}\u{A}", &["\u{300}", "\u{A}"]),
            ("\u{300}\u{308}\u{A}", &["\u{300}\u{308}", "\u{A}"]),
            ("\u{300}\u{1}", &["\u{300}", "\u{1}"]),
            ("\u{300}\u{308}\u{1}", &["\u{300}\u{308}", "\u{1}"]),
            ("\u{300}\u{300}", &["\u{300}\u{300}"]),
            ("\u{300}\u{308}\u{300}", &["\u{300}\u{308}\u{300}"]),
            ("\u{300}\u{1100}", &["\u{300}", "\u{1100}"]),
            ("\u{300}\u{308}\u{1100}", &["\u{300}\u{308}", "\u{1100}"]),
            ("\u{300}\u{1160}", &["\u{300}", "\u{1160}"]),
            ("\u{300}\u{308}\u{1160}", &["\u{300}\u{308}", "\u{1160}"]),
            ("\u{300}\u{11A8}", &["\u{300}", "\u{11A8}"]),
            ("\u{300}\u{308}\u{11A8}", &["\u{300}\u{308}", "\u{11A8}"]),
            ("\u{300}\u{AC00}", &["\u{300}", "\u{AC00}"]),
            ("\u{300}\u{308}\u{AC00}", &["\u{300}\u{308}", "\u{AC00}"]),
            ("\u{300}\u{AC01}", &["\u{300}", "\u{AC01}"]),
            ("\u{300}\u{308}\u{AC01}", &["\u{300}\u{308}", "\u{AC01}"]),
            ("\u{300}\u{1F1E6}", &["\u{300}", "\u{1F1E6}"]),
            ("\u{300}\u{308}\u{1F1E6}", &["\u{300}\u{308}", "\u{1F1E6}"]),
            ("\u{300}\u{378}", &["\u{300}", "\u{378}"]),
            ("\u{300}\u{308}\u{378}", &["\u{300}\u{308}", "\u{378}"]),
            ("\u{903}\u{20}", &["\u{903}", "\u{20}"]),
            ("\u{903}\u{308}\u{20}", &["\u{903}\u{308}", "\u{20}"]),
            ("\u{903}\u{D}", &["\u{903}", "\u{D}"]),
            ("\u{903}\u{308}\u{D}", &["\u{903}\u{308}", "\u{D}"]),
            ("\u{903}\u{A}", &["\u{903}", "\u{A}"]),
            ("\u{903}\u{308}\u{A}", &["\u{903}\u{308}", "\u{A}"]),
            ("\u{903}\u{1}", &["\u{903}", "\u{1}"]),
            ("\u{903}\u{308}\u{1}", &["\u{903}\u{308}", "\u{1}"]),
            ("\u{903}\u{300}", &["\u{903}\u{300}"]),
            ("\u{903}\u{308}\u{300}", &["\u{903}\u{308}\u{300}"]),
            ("\u{903}\u{1100}", &["\u{903}", "\u{1100}"]),
            ("\u{903}\u{308}\u{1100}", &["\u{903}\u{308}", "\u{1100}"]),
            ("\u{903}\u{1160}", &["\u{903}", "\u{1160}"]),
            ("\u{903}\u{308}\u{1160}", &["\u{903}\u{308}", "\u{1160}"]),
            ("\u{903}\u{11A8}", &["\u{903}", "\u{11A8}"]),
            ("\u{903}\u{308}\u{11A8}", &["\u{903}\u{308}", "\u{11A8}"]),
            ("\u{903}\u{AC00}", &["\u{903}", "\u{AC00}"]),
            ("\u{903}\u{308}\u{AC00}", &["\u{903}\u{308}", "\u{AC00}"]),
            ("\u{903}\u{AC01}", &["\u{903}", "\u{AC01}"]),
            ("\u{903}\u{308}\u{AC01}", &["\u{903}\u{308}", "\u{AC01}"]),
            ("\u{903}\u{1F1E6}", &["\u{903}", "\u{1F1E6}"]),
            ("\u{903}\u{308}\u{1F1E6}", &["\u{903}\u{308}", "\u{1F1E6}"]),
            ("\u{903}\u{378}", &["\u{903}", "\u{378}"]),
            ("\u{903}\u{308}\u{378}", &["\u{903}\u{308}", "\u{378}"]),
            ("\u{1100}\u{20}", &["\u{1100}", "\u{20}"]),
            ("\u{1100}\u{308}\u{20}", &["\u{1100}\u{308}", "\u{20}"]),
            ("\u{1100}\u{D}", &["\u{1100}", "\u{D}"]),
            ("\u{1100}\u{308}\u{D}", &["\u{1100}\u{308}", "\u{D}"]),
            ("\u{1100}\u{A}", &["\u{1100}", "\u{A}"]),
            ("\u{1100}\u{308}\u{A}", &["\u{1100}\u{308}", "\u{A}"]),
            ("\u{1100}\u{1}", &["\u{1100}", "\u{1}"]),
            ("\u{1100}\u{308}\u{1}", &["\u{1100}\u{308}", "\u{1}"]),
            ("\u{1100}\u{300}", &["\u{1100}\u{300}"]),
            ("\u{1100}\u{308}\u{300}", &["\u{1100}\u{308}\u{300}"]),
            ("\u{1100}\u{1100}", &["\u{1100}\u{1100}"]),
            ("\u{1100}\u{308}\u{1100}", &["\u{1100}\u{308}", "\u{1100}"]),
            ("\u{1100}\u{1160}", &["\u{1100}\u{1160}"]),
            ("\u{1100}\u{308}\u{1160}", &["\u{1100}\u{308}", "\u{1160}"]),
            ("\u{1100}\u{11A8}", &["\u{1100}", "\u{11A8}"]),
            ("\u{1100}\u{308}\u{11A8}", &["\u{1100}\u{308}", "\u{11A8}"]),
            ("\u{1100}\u{AC00}", &["\u{1100}\u{AC00}"]),
            ("\u{1100}\u{308}\u{AC00}", &["\u{1100}\u{308}", "\u{AC00}"]),
            ("\u{1100}\u{AC01}", &["\u{1100}\u{AC01}"]),
            ("\u{1100}\u{308}\u{AC01}", &["\u{1100}\u{308}", "\u{AC01}"]),
            ("\u{1100}\u{1F1E6}", &["\u{1100}", "\u{1F1E6}"]),
            ("\u{1100}\u{308}\u{1F1E6}", &["\u{1100}\u{308}", "\u{1F1E6}"]),
            ("\u{1100}\u{378}", &["\u{1100}", "\u{378}"]),
            ("\u{1100}\u{308}\u{378}", &["\u{1100}\u{308}", "\u{378}"]),
            ("\u{1160}\u{20}", &["\u{1160}", "\u{20}"]),
            ("\u{1160}\u{308}\u{20}", &["\u{1160}\u{308}", "\u{20}"]),
            ("\u{1160}\u{D}", &["\u{1160}", "\u{D}"]),
            ("\u{1160}\u{308}\u{D}", &["\u{1160}\u{308}", "\u{D}"]),
            ("\u{1160}\u{A}", &["\u{1160}", "\u{A}"]),
            ("\u{1160}\u{308}\u{A}", &["\u{1160}\u{308}", "\u{A}"]),
            ("\u{1160}\u{1}", &["\u{1160}", "\u{1}"]),
            ("\u{1160}\u{308}\u{1}", &["\u{1160}\u{308}", "\u{1}"]),
            ("\u{1160}\u{300}", &["\u{1160}\u{300}"]),
            ("\u{1160}\u{308}\u{300}", &["\u{1160}\u{308}\u{300}"]),
            ("\u{1160}\u{1100}", &["\u{1160}", "\u{1100}"]),
            ("\u{1160}\u{308}\u{1100}", &["\u{1160}\u{308}", "\u{1100}"]),
            ("\u{1160}\u{1160}", &["\u{1160}\u{1160}"]),
            ("\u{1160}\u{308}\u{1160}", &["\u{1160}\u{308}", "\u{1160}"]),
            ("\u{1160}\u{11A8}", &["\u{1160}\u{11A8}"]),
            ("\u{1160}\u{308}\u{11A8}", &["\u{1160}\u{308}", "\u{11A8}"]),
            ("\u{1160}\u{AC00}", &["\u{1160}", "\u{AC00}"]),
            ("\u{1160}\u{308}\u{AC00}", &["\u{1160}\u{308}", "\u{AC00}"]),
            ("\u{1160}\u{AC01}", &["\u{1160}", "\u{AC01}"]),
            ("\u{1160}\u{308}\u{AC01}", &["\u{1160}\u{308}", "\u{AC01}"]),
            ("\u{1160}\u{1F1E6}", &["\u{1160}", "\u{1F1E6}"]),
            ("\u{1160}\u{308}\u{1F1E6}", &["\u{1160}\u{308}", "\u{1F1E6}"]),
            ("\u{1160}\u{378}", &["\u{1160}", "\u{378}"]),
            ("\u{1160}\u{308}\u{378}", &["\u{1160}\u{308}", "\u{378}"]),
            ("\u{11A8}\u{20}", &["\u{11A8}", "\u{20}"]),
            ("\u{11A8}\u{308}\u{20}", &["\u{11A8}\u{308}", "\u{20}"]),
            ("\u{11A8}\u{D}", &["\u{11A8}", "\u{D}"]),
            ("\u{11A8}\u{308}\u{D}", &["\u{11A8}\u{308}", "\u{D}"]),
            ("\u{11A8}\u{A}", &["\u{11A8}", "\u{A}"]),
            ("\u{11A8}\u{308}\u{A}", &["\u{11A8}\u{308}", "\u{A}"]),
            ("\u{11A8}\u{1}", &["\u{11A8}", "\u{1}"]),
            ("\u{11A8}\u{308}\u{1}", &["\u{11A8}\u{308}", "\u{1}"]),
            ("\u{11A8}\u{300}", &["\u{11A8}\u{300}"]),
            ("\u{11A8}\u{308}\u{300}", &["\u{11A8}\u{308}\u{300}"]),
            ("\u{11A8}\u{1100}", &["\u{11A8}", "\u{1100}"]),
            ("\u{11A8}\u{308}\u{1100}", &["\u{11A8}\u{308}", "\u{1100}"]),
            ("\u{11A8}\u{1160}", &["\u{11A8}", "\u{1160}"]),
            ("\u{11A8}\u{308}\u{1160}", &["\u{11A8}\u{308}", "\u{1160}"]),
            ("\u{11A8}\u{11A8}", &["\u{11A8}\u{11A8}"]),
            ("\u{11A8}\u{308}\u{11A8}", &["\u{11A8}\u{308}", "\u{11A8}"]),
            ("\u{11A8}\u{AC00}", &["\u{11A8}", "\u{AC00}"]),
            ("\u{11A8}\u{308}\u{AC00}", &["\u{11A8}\u{308}", "\u{AC00}"]),
            ("\u{11A8}\u{AC01}", &["\u{11A8}", "\u{AC01}"]),
            ("\u{11A8}\u{308}\u{AC01}", &["\u{11A8}\u{308}", "\u{AC01}"]),
            ("\u{11A8}\u{1F1E6}", &["\u{11A8}", "\u{1F1E6}"]),
            ("\u{11A8}\u{308}\u{1F1E6}", &["\u{11A8}\u{308}", "\u{1F1E6}"]),
            ("\u{11A8}\u{378}", &["\u{11A8}", "\u{378}"]),
            ("\u{11A8}\u{308}\u{378}", &["\u{11A8}\u{308}", "\u{378}"]),
            ("\u{AC00}\u{20}", &["\u{AC00}", "\u{20}"]),
            ("\u{AC00}\u{308}\u{20}", &["\u{AC00}\u{308}", "\u{20}"]),
            ("\u{AC00}\u{D}", &["\u{AC00}", "\u{D}"]),
            ("\u{AC00}\u{308}\u{D}", &["\u{AC00}\u{308}", "\u{D}"]),
            ("\u{AC00}\u{A}", &["\u{AC00}", "\u{A}"]),
            ("\u{AC00}\u{308}\u{A}", &["\u{AC00}\u{308}", "\u{A}"]),
            ("\u{AC00}\u{1}", &["\u{AC00}", "\u{1}"]),
            ("\u{AC00}\u{308}\u{1}", &["\u{AC00}\u{308}", "\u{1}"]),
            ("\u{AC00}\u{300}", &["\u{AC00}\u{300}"]),
            ("\u{AC00}\u{308}\u{300}", &["\u{AC00}\u{308}\u{300}"]),
            ("\u{AC00}\u{1100}", &["\u{AC00}", "\u{1100}"]),
            ("\u{AC00}\u{308}\u{1100}", &["\u{AC00}\u{308}", "\u{1100}"]),
            ("\u{AC00}\u{1160}", &["\u{AC00}\u{1160}"]),
            ("\u{AC00}\u{308}\u{1160}", &["\u{AC00}\u{308}", "\u{1160}"]),
            ("\u{AC00}\u{11A8}", &["\u{AC00}\u{11A8}"]),
            ("\u{AC00}\u{308}\u{11A8}", &["\u{AC00}\u{308}", "\u{11A8}"]),
            ("\u{AC00}\u{AC00}", &["\u{AC00}", "\u{AC00}"]),
            ("\u{AC00}\u{308}\u{AC00}", &["\u{AC00}\u{308}", "\u{AC00}"]),
            ("\u{AC00}\u{AC01}", &["\u{AC00}", "\u{AC01}"]),
            ("\u{AC00}\u{308}\u{AC01}", &["\u{AC00}\u{308}", "\u{AC01}"]),
            ("\u{AC00}\u{1F1E6}", &["\u{AC00}", "\u{1F1E6}"]),
            ("\u{AC00}\u{308}\u{1F1E6}", &["\u{AC00}\u{308}", "\u{1F1E6}"]),
            ("\u{AC00}\u{378}", &["\u{AC00}", "\u{378}"]),
            ("\u{AC00}\u{308}\u{378}", &["\u{AC00}\u{308}", "\u{378}"]),
            ("\u{AC01}\u{20}", &["\u{AC01}", "\u{20}"]),
            ("\u{AC01}\u{308}\u{20}", &["\u{AC01}\u{308}", "\u{20}"]),
            ("\u{AC01}\u{D}", &["\u{AC01}", "\u{D}"]),
            ("\u{AC01}\u{308}\u{D}", &["\u{AC01}\u{308}", "\u{D}"]),
            ("\u{AC01}\u{A}", &["\u{AC01}", "\u{A}"]),
            ("\u{AC01}\u{308}\u{A}", &["\u{AC01}\u{308}", "\u{A}"]),
            ("\u{AC01}\u{1}", &["\u{AC01}", "\u{1}"]),
            ("\u{AC01}\u{308}\u{1}", &["\u{AC01}\u{308}", "\u{1}"]),
            ("\u{AC01}\u{300}", &["\u{AC01}\u{300}"]),
            ("\u{AC01}\u{308}\u{300}", &["\u{AC01}\u{308}\u{300}"]),
            ("\u{AC01}\u{1100}", &["\u{AC01}", "\u{1100}"]),
            ("\u{AC01}\u{308}\u{1100}", &["\u{AC01}\u{308}", "\u{1100}"]),
            ("\u{AC01}\u{1160}", &["\u{AC01}", "\u{1160}"]),
            ("\u{AC01}\u{308}\u{1160}", &["\u{AC01}\u{308}", "\u{1160}"]),
            ("\u{AC01}\u{11A8}", &["\u{AC01}\u{11A8}"]),
            ("\u{AC01}\u{308}\u{11A8}", &["\u{AC01}\u{308}", "\u{11A8}"]),
            ("\u{AC01}\u{AC00}", &["\u{AC01}", "\u{AC00}"]),
            ("\u{AC01}\u{308}\u{AC00}", &["\u{AC01}\u{308}", "\u{AC00}"]),
            ("\u{AC01}\u{AC01}", &["\u{AC01}", "\u{AC01}"]),
            ("\u{AC01}\u{308}\u{AC01}", &["\u{AC01}\u{308}", "\u{AC01}"]),
            ("\u{AC01}\u{1F1E6}", &["\u{AC01}", "\u{1F1E6}"]),
            ("\u{AC01}\u{308}\u{1F1E6}", &["\u{AC01}\u{308}", "\u{1F1E6}"]),
            ("\u{AC01}\u{378}", &["\u{AC01}", "\u{378}"]),
            ("\u{AC01}\u{308}\u{378}", &["\u{AC01}\u{308}", "\u{378}"]),
            ("\u{1F1E6}\u{20}", &["\u{1F1E6}", "\u{20}"]),
            ("\u{1F1E6}\u{308}\u{20}", &["\u{1F1E6}\u{308}", "\u{20}"]),
            ("\u{1F1E6}\u{D}", &["\u{1F1E6}", "\u{D}"]),
            ("\u{1F1E6}\u{308}\u{D}", &["\u{1F1E6}\u{308}", "\u{D}"]),
            ("\u{1F1E6}\u{A}", &["\u{1F1E6}", "\u{A}"]),
            ("\u{1F1E6}\u{308}\u{A}", &["\u{1F1E6}\u{308}", "\u{A}"]),
            ("\u{1F1E6}\u{1}", &["\u{1F1E6}", "\u{1}"]),
            ("\u{1F1E6}\u{308}\u{1}", &["\u{1F1E6}\u{308}", "\u{1}"]),
            ("\u{1F1E6}\u{300}", &["\u{1F1E6}\u{300}"]),
            ("\u{1F1E6}\u{308}\u{300}", &["\u{1F1E6}\u{308}\u{300}"]),
            ("\u{1F1E6}\u{1100}", &["\u{1F1E6}", "\u{1100}"]),
            ("\u{1F1E6}\u{308}\u{1100}", &["\u{1F1E6}\u{308}", "\u{1100}"]),
            ("\u{1F1E6}\u{1160}", &["\u{1F1E6}", "\u{1160}"]),
            ("\u{1F1E6}\u{308}\u{1160}", &["\u{1F1E6}\u{308}", "\u{1160}"]),
            ("\u{1F1E6}\u{11A8}", &["\u{1F1E6}", "\u{11A8}"]),
            ("\u{1F1E6}\u{308}\u{11A8}", &["\u{1F1E6}\u{308}", "\u{11A8}"]),
            ("\u{1F1E6}\u{AC00}", &["\u{1F1E6}", "\u{AC00}"]),
            ("\u{1F1E6}\u{308}\u{AC00}", &["\u{1F1E6}\u{308}", "\u{AC00}"]),
            ("\u{1F1E6}\u{AC01}", &["\u{1F1E6}", "\u{AC01}"]),
            ("\u{1F1E6}\u{308}\u{AC01}", &["\u{1F1E6}\u{308}", "\u{AC01}"]),
            ("\u{1F1E6}\u{1F1E6}", &["\u{1F1E6}\u{1F1E6}"]),
            ("\u{1F1E6}\u{308}\u{1F1E6}", &["\u{1F1E6}\u{308}", "\u{1F1E6}"]),
            ("\u{1F1E6}\u{378}", &["\u{1F1E6}", "\u{378}"]),
            ("\u{1F1E6}\u{308}\u{378}", &["\u{1F1E6}\u{308}", "\u{378}"]),
            ("\u{378}\u{20}", &["\u{378}", "\u{20}"]),
            ("\u{378}\u{308}\u{20}", &["\u{378}\u{308}", "\u{20}"]),
            ("\u{378}\u{D}", &["\u{378}", "\u{D}"]),
            ("\u{378}\u{308}\u{D}", &["\u{378}\u{308}", "\u{D}"]),
            ("\u{378}\u{A}", &["\u{378}", "\u{A}"]),
            ("\u{378}\u{308}\u{A}", &["\u{378}\u{308}", "\u{A}"]),
            ("\u{378}\u{1}", &["\u{378}", "\u{1}"]),
            ("\u{378}\u{308}\u{1}", &["\u{378}\u{308}", "\u{1}"]),
            ("\u{378}\u{300}", &["\u{378}\u{300}"]),
            ("\u{378}\u{308}\u{300}", &["\u{378}\u{308}\u{300}"]),
            ("\u{378}\u{1100}", &["\u{378}", "\u{1100}"]),
            ("\u{378}\u{308}\u{1100}", &["\u{378}\u{308}", "\u{1100}"]),
            ("\u{378}\u{1160}", &["\u{378}", "\u{1160}"]),
            ("\u{378}\u{308}\u{1160}", &["\u{378}\u{308}", "\u{1160}"]),
            ("\u{378}\u{11A8}", &["\u{378}", "\u{11A8}"]),
            ("\u{378}\u{308}\u{11A8}", &["\u{378}\u{308}", "\u{11A8}"]),
            ("\u{378}\u{AC00}", &["\u{378}", "\u{AC00}"]),
            ("\u{378}\u{308}\u{AC00}", &["\u{378}\u{308}", "\u{AC00}"]),
            ("\u{378}\u{AC01}", &["\u{378}", "\u{AC01}"]),
            ("\u{378}\u{308}\u{AC01}", &["\u{378}\u{308}", "\u{AC01}"]),
            ("\u{378}\u{1F1E6}", &["\u{378}", "\u{1F1E6}"]),
            ("\u{378}\u{308}\u{1F1E6}", &["\u{378}\u{308}", "\u{1F1E6}"]),
            ("\u{378}\u{378}", &["\u{378}", "\u{378}"]),
            ("\u{378}\u{308}\u{378}", &["\u{378}\u{308}", "\u{378}"]),
            ("\u{61}\u{1F1E6}\u{62}", &["\u{61}", "\u{1F1E6}", "\u{62}"]),
            ("\u{1F1F7}\u{1F1FA}", &["\u{1F1F7}\u{1F1FA}"]),
            ("\u{1F1F7}\u{1F1FA}\u{1F1F8}", &["\u{1F1F7}\u{1F1FA}\u{1F1F8}"]),
            ("\u{1F1F7}\u{1F1FA}\u{1F1F8}\u{1F1EA}",
            &["\u{1F1F7}\u{1F1FA}\u{1F1F8}\u{1F1EA}"]),
            ("\u{1F1F7}\u{1F1FA}\u{200B}\u{1F1F8}\u{1F1EA}",
             &["\u{1F1F7}\u{1F1FA}", "\u{200B}", "\u{1F1F8}\u{1F1EA}"]),
            ("\u{1F1E6}\u{1F1E7}\u{1F1E8}", &["\u{1F1E6}\u{1F1E7}\u{1F1E8}"]),
            ("\u{1F1E6}\u{200D}\u{1F1E7}\u{1F1E8}", &["\u{1F1E6}\u{200D}",
             "\u{1F1E7}\u{1F1E8}"]),
            ("\u{1F1E6}\u{1F1E7}\u{200D}\u{1F1E8}",
             &["\u{1F1E6}\u{1F1E7}\u{200D}", "\u{1F1E8}"]),
            ("\u{20}\u{200D}\u{646}", &["\u{20}\u{200D}", "\u{646}"]),
            ("\u{646}\u{200D}\u{20}", &["\u{646}\u{200D}", "\u{20}"]),
        ];

        let test_diff: [(_, &[_], &[_]); 23] = [
            ("\u{20}\u{903}", &["\u{20}\u{903}"], &["\u{20}", "\u{903}"]), ("\u{20}\u{308}\u{903}",
            &["\u{20}\u{308}\u{903}"], &["\u{20}\u{308}", "\u{903}"]), ("\u{D}\u{308}\u{903}",
            &["\u{D}", "\u{308}\u{903}"], &["\u{D}", "\u{308}", "\u{903}"]), ("\u{A}\u{308}\u{903}",
            &["\u{A}", "\u{308}\u{903}"], &["\u{A}", "\u{308}", "\u{903}"]), ("\u{1}\u{308}\u{903}",
            &["\u{1}", "\u{308}\u{903}"], &["\u{1}", "\u{308}", "\u{903}"]), ("\u{300}\u{903}",
            &["\u{300}\u{903}"], &["\u{300}", "\u{903}"]), ("\u{300}\u{308}\u{903}",
            &["\u{300}\u{308}\u{903}"], &["\u{300}\u{308}", "\u{903}"]), ("\u{903}\u{903}",
            &["\u{903}\u{903}"], &["\u{903}", "\u{903}"]), ("\u{903}\u{308}\u{903}",
            &["\u{903}\u{308}\u{903}"], &["\u{903}\u{308}", "\u{903}"]), ("\u{1100}\u{903}",
            &["\u{1100}\u{903}"], &["\u{1100}", "\u{903}"]), ("\u{1100}\u{308}\u{903}",
            &["\u{1100}\u{308}\u{903}"], &["\u{1100}\u{308}", "\u{903}"]), ("\u{1160}\u{903}",
            &["\u{1160}\u{903}"], &["\u{1160}", "\u{903}"]), ("\u{1160}\u{308}\u{903}",
            &["\u{1160}\u{308}\u{903}"], &["\u{1160}\u{308}", "\u{903}"]), ("\u{11A8}\u{903}",
            &["\u{11A8}\u{903}"], &["\u{11A8}", "\u{903}"]), ("\u{11A8}\u{308}\u{903}",
            &["\u{11A8}\u{308}\u{903}"], &["\u{11A8}\u{308}", "\u{903}"]), ("\u{AC00}\u{903}",
            &["\u{AC00}\u{903}"], &["\u{AC00}", "\u{903}"]), ("\u{AC00}\u{308}\u{903}",
            &["\u{AC00}\u{308}\u{903}"], &["\u{AC00}\u{308}", "\u{903}"]), ("\u{AC01}\u{903}",
            &["\u{AC01}\u{903}"], &["\u{AC01}", "\u{903}"]), ("\u{AC01}\u{308}\u{903}",
            &["\u{AC01}\u{308}\u{903}"], &["\u{AC01}\u{308}", "\u{903}"]), ("\u{1F1E6}\u{903}",
            &["\u{1F1E6}\u{903}"], &["\u{1F1E6}", "\u{903}"]), ("\u{1F1E6}\u{308}\u{903}",
            &["\u{1F1E6}\u{308}\u{903}"], &["\u{1F1E6}\u{308}", "\u{903}"]), ("\u{378}\u{903}",
            &["\u{378}\u{903}"], &["\u{378}", "\u{903}"]), ("\u{378}\u{308}\u{903}",
            &["\u{378}\u{308}\u{903}"], &["\u{378}\u{308}", "\u{903}"]),
        ];

        for &(s, g) in &test_same[..] {
            // test forward iterator
            assert!(order::equals(s.graphemes(true), g.iter().cloned()));
            assert!(order::equals(s.graphemes(false), g.iter().cloned()));

            // test reverse iterator
            assert!(order::equals(s.graphemes(true).rev(), g.iter().rev().cloned()));
            assert!(order::equals(s.graphemes(false).rev(), g.iter().rev().cloned()));
        }

        for &(s, gt, gf) in &test_diff {
            // test forward iterator
            assert!(order::equals(s.graphemes(true), gt.iter().cloned()));
            assert!(order::equals(s.graphemes(false), gf.iter().cloned()));

            // test reverse iterator
            assert!(order::equals(s.graphemes(true).rev(), gt.iter().rev().cloned()));
            assert!(order::equals(s.graphemes(false).rev(), gf.iter().rev().cloned()));
        }

        // test the indices iterators
        let s = "a̐éö̲\r\n";
        let gr_inds = s.grapheme_indices(true).collect::<Vec<(usize, &str)>>();
        let b: &[_] = &[(0, "a̐"), (3, "é"), (6, "ö̲"), (11, "\r\n")];
        assert_eq!(gr_inds, b);
        let gr_inds = s.grapheme_indices(true).rev().collect::<Vec<(usize, &str)>>();
        let b: &[_] = &[(11, "\r\n"), (6, "ö̲"), (3, "é"), (0, "a̐")];
        assert_eq!(gr_inds, b);
        let mut gr_inds_iter = s.grapheme_indices(true);
        {
            let gr_inds = gr_inds_iter.by_ref();
            let e1 = gr_inds.size_hint();
            assert_eq!(e1, (1, Some(13)));
            let c = gr_inds.count();
            assert_eq!(c, 4);
        }
        let e2 = gr_inds_iter.size_hint();
        assert_eq!(e2, (0, Some(0)));

        // make sure the reverse iterator does the right thing with "\n" at beginning of string
        let s = "\n\r\n\r";
        let gr = s.graphemes(true).rev().collect::<Vec<&str>>();
        let b: &[_] = &["\r", "\r\n", "\n"];
        assert_eq!(gr, b);
    }

    #[test]
    fn test_split_strator() {
        fn t(s: &str, sep: &str, u: &[&str]) {
            let v: Vec<&str> = s.split_str(sep).collect();
            assert_eq!(v, u);
        }
        t("--1233345--", "12345", &["--1233345--"]);
        t("abc::hello::there", "::", &["abc", "hello", "there"]);
        t("::hello::there", "::", &["", "hello", "there"]);
        t("hello::there::", "::", &["hello", "there", ""]);
        t("::hello::there::", "::", &["", "hello", "there", ""]);
        t("ประเทศไทย中华Việt Nam", "中华", &["ประเทศไทย", "Việt Nam"]);
        t("zzXXXzzYYYzz", "zz", &["", "XXX", "YYY", ""]);
        t("zzXXXzYYYz", "XXX", &["zz", "zYYYz"]);
        t(".XXX.YYY.", ".", &["", "XXX", "YYY", ""]);
        t("", ".", &[""]);
        t("zz", "zz", &["",""]);
        t("ok", "z", &["ok"]);
        t("zzz", "zz", &["","z"]);
        t("zzzzz", "zz", &["","","z"]);
    }

    #[test]
    fn test_str_default() {
        use core::default::Default;
        fn t<S: Default + Str>() {
            let s: S = Default::default();
            assert_eq!(s.as_slice(), "");
        }

        t::<&str>();
        t::<String>();
    }

    #[test]
    fn test_str_container() {
        fn sum_len(v: &[&str]) -> usize {
            v.iter().map(|x| x.len()).sum()
        }

        let s = String::from_str("01234");
        assert_eq!(5, sum_len(&["012", "", "34"]));
        assert_eq!(5, sum_len(&[&String::from_str("01"),
                                &String::from_str("2"),
                                &String::from_str("34"),
                                &String::from_str("")]));
        assert_eq!(5, sum_len(&[&s]));
    }

    #[test]
    fn test_str_from_utf8() {
        let xs = b"hello";
        assert_eq!(from_utf8(xs), Ok("hello"));

        let xs = "ศไทย中华Việt Nam".as_bytes();
        assert_eq!(from_utf8(xs), Ok("ศไทย中华Việt Nam"));

        let xs = b"hello\xFF";
        assert_eq!(from_utf8(xs), Err(Utf8Error::TooShort));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use prelude::{SliceExt, IteratorExt, SliceConcatExt};
    use test::Bencher;
    use test::black_box;

    #[bench]
    fn char_iterator(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().count());
    }

    #[bench]
    fn char_iterator_for(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| {
            for ch in s.chars() { black_box(ch); }
        });
    }

    #[bench]
    fn char_iterator_ascii(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().count());
    }

    #[bench]
    fn char_iterator_rev(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().rev().count());
    }

    #[bench]
    fn char_iterator_rev_for(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| {
            for ch in s.chars().rev() { black_box(ch); }
        });
    }

    #[bench]
    fn char_indicesator(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.chars().count();

        b.iter(|| assert_eq!(s.char_indices().count(), len));
    }

    #[bench]
    fn char_indicesator_rev(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.chars().count();

        b.iter(|| assert_eq!(s.char_indices().rev().count(), len));
    }

    #[bench]
    fn split_unicode_ascii(b: &mut Bencher) {
        let s = "ประเทศไทย中华Việt Namประเทศไทย中华Việt Nam";

        b.iter(|| assert_eq!(s.split('V').count(), 3));
    }

    #[bench]
    fn split_unicode_not_ascii(b: &mut Bencher) {
        struct NotAscii(char);
        impl CharEq for NotAscii {
            fn matches(&mut self, c: char) -> bool {
                let NotAscii(cc) = *self;
                cc == c
            }
            fn only_ascii(&self) -> bool { false }
        }
        let s = "ประเทศไทย中华Việt Namประเทศไทย中华Việt Nam";

        b.iter(|| assert_eq!(s.split(NotAscii('V')).count(), 3));
    }


    #[bench]
    fn split_ascii(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        b.iter(|| assert_eq!(s.split(' ').count(), len));
    }

    #[bench]
    fn split_not_ascii(b: &mut Bencher) {
        struct NotAscii(char);
        impl CharEq for NotAscii {
            #[inline]
            fn matches(&mut self, c: char) -> bool {
                let NotAscii(cc) = *self;
                cc == c
            }
            fn only_ascii(&self) -> bool { false }
        }
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        b.iter(|| assert_eq!(s.split(NotAscii(' ')).count(), len));
    }

    #[bench]
    fn split_extern_fn(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();
        fn pred(c: char) -> bool { c == ' ' }

        b.iter(|| assert_eq!(s.split(pred).count(), len));
    }

    #[bench]
    fn split_closure(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        b.iter(|| assert_eq!(s.split(|c: char| c == ' ').count(), len));
    }

    #[bench]
    fn split_slice(b: &mut Bencher) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').count();

        let c: &[char] = &[' '];
        b.iter(|| assert_eq!(s.split(c).count(), len));
    }

    #[bench]
    fn bench_connect(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let sep = "→";
        let v = vec![s, s, s, s, s, s, s, s, s, s];
        b.iter(|| {
            assert_eq!(v.connect(sep).len(), s.len() * 10 + sep.len() * 9);
        })
    }

    #[bench]
    fn bench_contains_short_short(b: &mut Bencher) {
        let haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let needle = "sit";

        b.iter(|| {
            assert!(haystack.contains(needle));
        })
    }

    #[bench]
    fn bench_contains_short_long(b: &mut Bencher) {
        let haystack = "\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem sit amet dolor \
ultricies condimentum. Praesent iaculis purus elit, ac malesuada quam malesuada in. Duis sed orci \
eros. Suspendisse sit amet magna mollis, mollis nunc luctus, imperdiet mi. Integer fringilla non \
sem ut lacinia. Fusce varius tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec \
tempus vel, gravida nec quam.

In est dui, tincidunt sed tempus interdum, adipiscing laoreet ante. Etiam tempor, tellus quis \
sagittis interdum, nulla purus mattis sem, quis auctor erat odio ac tellus. In nec nunc sit amet \
diam volutpat molestie at sed ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan \
lorem ac dignissim placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat. In vel \
eleifend felis. Sed suscipit nulla lorem, sed mollis est sollicitudin et. Nam fermentum egestas \
interdum. Curabitur ut nisi justo.

Sed sollicitudin ipsum tellus, ut condimentum leo eleifend nec. Cras ut velit ante. Phasellus nec \
mollis odio. Mauris molestie erat in arcu mattis, at aliquet dolor vehicula. Quisque malesuada \
lectus sit amet nisi pretium, a condimentum ipsum porta. Morbi at dapibus diam. Praesent egestas \
est sed risus elementum, eu rutrum metus ultrices. Etiam fermentum consectetur magna, id rutrum \
felis accumsan a. Aliquam ut pellentesque libero. Sed mi nulla, lobortis eu tortor id, suscipit \
ultricies neque. Morbi iaculis sit amet risus at iaculis. Praesent eget ligula quis turpis \
feugiat suscipit vel non arcu. Interdum et malesuada fames ac ante ipsum primis in faucibus. \
Aliquam sit amet placerat lorem.

Cras a lacus vel ante posuere elementum. Nunc est leo, bibendum ut facilisis vel, bibendum at \
mauris. Nullam adipiscing diam vel odio ornare, luctus adipiscing mi luctus. Nulla facilisi. \
Mauris adipiscing bibendum neque, quis adipiscing lectus tempus et. Sed feugiat erat et nisl \
lobortis pharetra. Donec vitae erat enim. Nullam sit amet felis et quam lacinia tincidunt. Aliquam \
suscipit dapibus urna. Sed volutpat urna in magna pulvinar volutpat. Phasellus nec tellus ac diam \
cursus accumsan.

Nam lectus enim, dapibus non nisi tempor, consectetur convallis massa. Maecenas eleifend dictum \
feugiat. Etiam quis mauris vel risus luctus mattis a a nunc. Nullam orci quam, imperdiet id \
vehicula in, porttitor ut nibh. Duis sagittis adipiscing nisl vitae congue. Donec mollis risus eu \
leo suscipit, varius porttitor nulla porta. Pellentesque ut sem nec nisi euismod vehicula. Nulla \
malesuada sollicitudin quam eu fermentum.";
        let needle = "english";

        b.iter(|| {
            assert!(!haystack.contains(needle));
        })
    }

    #[bench]
    fn bench_contains_bad_naive(b: &mut Bencher) {
        let haystack = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let needle = "aaaaaaaab";

        b.iter(|| {
            assert!(!haystack.contains(needle));
        })
    }

    #[bench]
    fn bench_contains_equal(b: &mut Bencher) {
        let haystack = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let needle = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";

        b.iter(|| {
            assert!(haystack.contains(needle));
        })
    }
}
