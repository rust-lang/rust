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

//! Unicode string manipulation (the `str` type).
//!
//! Rust's `str` type is one of the core primitive types of the language. `&str`
//! is the borrowed string type. This type of string can only be created from
//! other strings, unless it is a `&'static str` (see below). It is not possible
//! to move out of borrowed strings because they are owned elsewhere.
//!
//! # Examples
//!
//! Here's some code that uses a `&str`:
//!
//! ```
//! let s = "Hello, world.";
//! ```
//!
//! This `&str` is a `&'static str`, which is the type of string literals.
//! They're `'static` because literals are available for the entire lifetime of
//! the program.
//!
//! You can get a non-`'static` `&str` by taking a slice of a `String`:
//!
//! ```
//! # let some_string = "Hello, world.".to_string();
//! let s = &some_string;
//! ```
//!
//! # Representation
//!
//! Rust's string type, `str`, is a sequence of Unicode scalar values encoded as
//! a stream of UTF-8 bytes. All [strings](../../reference.html#literals) are
//! guaranteed to be validly encoded UTF-8 sequences. Additionally, strings are
//! not null-terminated and can thus contain null bytes.
//!
//! The actual representation of `str`s have direct mappings to slices: `&str`
//! is the same as `&[u8]`.

#![doc(primitive = "str")]
#![stable(feature = "rust1", since = "1.0.0")]

use self::RecompositionState::*;
use self::DecompositionType::*;

use core::clone::Clone;
use core::iter::{Iterator, Extend};
use core::option::Option::{self, Some, None};
use core::result::Result;
use core::str as core_str;
use core::str::pattern::Pattern;
use core::str::pattern::{Searcher, ReverseSearcher, DoubleEndedSearcher};
use unicode::str::{UnicodeStr, Utf16Encoder};

use core::convert::AsRef;
use vec_deque::VecDeque;
use borrow::{Borrow, ToOwned};
use string::String;
use unicode;
use vec::Vec;
use slice::SliceConcatExt;

pub use core::str::{FromStr, Utf8Error, Str};
pub use core::str::{Lines, LinesAny, CharRange};
pub use core::str::{Split, RSplit};
pub use core::str::{SplitN, RSplitN};
pub use core::str::{SplitTerminator, RSplitTerminator};
pub use core::str::{Matches, RMatches};
pub use core::str::{MatchIndices, RMatchIndices};
pub use core::str::{from_utf8, Chars, CharIndices, Bytes};
pub use core::str::{from_utf8_unchecked, ParseBoolError};
pub use unicode::str::{Words, Graphemes, GraphemeIndices};
pub use core::str::pattern;

/*
Section: Creating a string
*/

impl<S: AsRef<str>> SliceConcatExt<str, String> for [S] {
    fn concat(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        // `len` calculation may overflow but push_str will check boundaries
        let len = self.iter().map(|s| s.as_ref().len()).sum();
        let mut result = String::with_capacity(len);

        for s in self {
            result.push_str(s.as_ref())
        }

        result
    }

    fn connect(&self, sep: &str) -> String {
        if self.is_empty() {
            return String::new();
        }

        // concat is faster
        if sep.is_empty() {
            return self.concat();
        }

        // this is wrong without the guarantee that `self` is non-empty
        // `len` calculation may overflow but push_str but will check boundaries
        let len = sep.len() * (self.len() - 1)
            + self.iter().map(|s| s.as_ref().len()).sum::<usize>();
        let mut result = String::with_capacity(len);
        let mut first = true;

        for s in self {
            if first {
                first = false;
            } else {
                result.push_str(sep);
            }
            result.push_str(s.as_ref());
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

/// External iterator for a string decomposition's characters.
///
/// For use with the `std::iter` module.
#[derive(Clone)]
#[unstable(feature = "unicode",
           reason = "this functionality may be replaced with a more generic \
                     unicode crate on crates.io")]
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

/// External iterator for a string recomposition's characters.
///
/// For use with the `std::iter` module.
#[derive(Clone)]
#[unstable(feature = "unicode",
           reason = "this functionality may be replaced with a more generic \
                     unicode crate on crates.io")]
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
///
/// For use with the `std::iter` module.
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
    ($ch:expr, $byte:expr) => (($ch << 6) | ($byte & 63) as u32)
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
#[lang = "str"]
#[cfg(not(test))]
#[stable(feature = "rust1", since = "1.0.0")]
impl str {
    /// Escapes each char in `s` with `char::escape_default`.
    #[unstable(feature = "collections",
               reason = "return type may change to be an iterator")]
    pub fn escape_default(&self) -> String {
        self.chars().flat_map(|c| c.escape_default()).collect()
    }

    /// Escapes each char in `s` with `char::escape_unicode`.
    #[unstable(feature = "collections",
               reason = "return type may change to be an iterator")]
    pub fn escape_unicode(&self) -> String {
        self.chars().flat_map(|c| c.escape_unicode()).collect()
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
    #[unstable(feature = "unicode",
               reason = "this functionality may be replaced with a more generic \
                         unicode crate on crates.io")]
    pub fn nfd_chars(&self) -> Decompositions {
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
    #[unstable(feature = "unicode",
               reason = "this functionality may be replaced with a more generic \
                         unicode crate on crates.io")]
    pub fn nfkd_chars(&self) -> Decompositions {
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
    #[unstable(feature = "unicode",
               reason = "this functionality may be replaced with a more generic \
                         unicode crate on crates.io")]
    pub fn nfc_chars(&self) -> Recompositions {
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
    #[unstable(feature = "unicode",
               reason = "this functionality may be replaced with a more generic \
                         unicode crate on crates.io")]
    pub fn nfkc_chars(&self) -> Recompositions {
        Recompositions {
            iter: self.nfkd_chars(),
            state: Composing,
            buffer: VecDeque::new(),
            composee: None,
            last_ccc: None
        }
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
        core_str::StrExt::contains(&self[..], pat)
    }

    /// An iterator over the codepoints of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<char> = "abc åäö".chars().collect();
    ///
    /// assert_eq!(v, ['a', 'b', 'c', ' ', 'å', 'ä', 'ö']);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn chars(&self) -> Chars {
        core_str::StrExt::chars(&self[..])
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
    pub fn bytes(&self) -> Bytes {
        core_str::StrExt::bytes(&self[..])
    }

    /// An iterator over the characters of `self` and their byte offsets.
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<(usize, char)> = "abc".char_indices().collect();
    /// let b = vec![(0, 'a'), (1, 'b'), (2, 'c')];
    ///
    /// assert_eq!(v, b);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn char_indices(&self) -> CharIndices {
        core_str::StrExt::char_indices(&self[..])
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
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
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1def2ghi".split(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    ///
    /// let v: Vec<&str> = "lionXtigerXleopard".split(char::is_uppercase).collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P> {
        core_str::StrExt::split(&self[..], pat)
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
    /// More complex patterns with closures:
    ///
    /// ```rust
    /// let v: Vec<&str> = "abc1def2ghi".rsplit(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["ghi", "def", "abc"]);
    ///
    /// let v: Vec<&str> = "lionXtigerXleopard".rsplit(char::is_uppercase).collect();
    /// assert_eq!(v, ["leopard", "tiger", "lion"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rsplit<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplit<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplit(&self[..], pat)
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
    /// rather than _seperated_ by a pattern.
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
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".split_terminator('.').collect();
    /// assert_eq!(v, ["A", "B"]);
    ///
    /// let v: Vec<&str> = "A..B..".split_terminator(".").collect();
    /// assert_eq!(v, ["A", "", "B", ""]);
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1def2ghi3".split_terminator(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P> {
        core_str::StrExt::split_terminator(&self[..], pat)
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
    /// rather than _seperated_ by a pattern.
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
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".rsplit_terminator('.').collect();
    /// assert_eq!(v, ["B", "A"]);
    ///
    /// let v: Vec<&str> = "A..B..".rsplit_terminator(".").collect();
    /// assert_eq!(v, ["", "B", "", "A"]);
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1def2ghi3".rsplit_terminator(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["ghi", "def", "abc"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rsplit_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplitTerminator<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplit_terminator(&self[..], pat)
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
    /// More complex patterns with closures:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1def2ghi".splitn(2, |c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["abc", "def2ghi"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn splitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> SplitN<'a, P> {
        core_str::StrExt::splitn(&self[..], count, pat)
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
    /// More complex patterns with closures:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1def2ghi".rsplitn(2, |c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["ghi", "abc1def"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn rsplitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> RSplitN<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rsplitn(&self[..], count, pat)
    }

    /// An iterator over the matches of a pattern within `self`.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
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
    /// # #![feature(collections)]
    /// let v: Vec<&str> = "abcXXXabcYYYabc".matches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".matches(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["1", "2", "3"]);
    /// ```
    #[unstable(feature = "collections",
               reason = "method got recently added")]
    pub fn matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> Matches<'a, P> {
        core_str::StrExt::matches(&self[..], pat)
    }

    /// An iterator over the matches of a pattern within `self`, yielded in
    /// reverse order.
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
    /// For iterating from the front, `matches()` can be used.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// let v: Vec<&str> = "abcXXXabcYYYabc".rmatches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".rmatches(|c: char| c.is_numeric()).collect();
    /// assert_eq!(v, ["3", "2", "1"]);
    /// ```
    #[unstable(feature = "collections",
               reason = "method got recently added")]
    pub fn rmatches<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatches<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rmatches(&self[..], pat)
    }

    /// An iterator over the start and end indices of the disjoint matches
    /// of a pattern within `self`.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the first
    /// match are returned.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines
    /// the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be double ended if the pattern allows a
    /// reverse search
    /// and forward/reverse search yields the same elements. This is true for,
    /// eg, `char` but not
    /// for `&str`.
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, `rmatch_indices()` can be used.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// let v: Vec<(usize, usize)> = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, [(0, 3), (6, 9), (12, 15)]);
    ///
    /// let v: Vec<(usize, usize)> = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, [(1, 4), (4, 7)]);
    ///
    /// let v: Vec<(usize, usize)> = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, [(0, 3)]); // only the first `aba`
    /// ```
    #[unstable(feature = "collections",
               reason = "might have its iterator type changed")]
    // NB: Right now MatchIndices yields `(usize, usize)`, but it would
    // be more consistent with `matches` and `char_indices` to return `(usize, &str)`
    pub fn match_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> MatchIndices<'a, P> {
        core_str::StrExt::match_indices(&self[..], pat)
    }

    /// An iterator over the start and end indices of the disjoint matches of
    /// a pattern within
    /// `self`, yielded in reverse order.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the last
    /// match are returned.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines
    /// the split.
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
    /// For iterating from the front, `match_indices()` can be used.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// let v: Vec<(usize, usize)> = "abcXXXabcYYYabc".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(12, 15), (6, 9), (0, 3)]);
    ///
    /// let v: Vec<(usize, usize)> = "1abcabc2".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(4, 7), (1, 4)]);
    ///
    /// let v: Vec<(usize, usize)> = "ababa".rmatch_indices("aba").collect();
    /// assert_eq!(v, [(2, 5)]); // only the last `aba`
    /// ```
    #[unstable(feature = "collections",
               reason = "might have its iterator type changed")]
    // NB: Right now RMatchIndices yields `(usize, usize)`, but it would
    // be more consistent with `rmatches` and `char_indices` to return `(usize, &str)`
    pub fn rmatch_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatchIndices<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::rmatch_indices(&self[..], pat)
    }

    /// An iterator over the lines of a string, separated by `\n`.
    ///
    /// This does not include the empty string after a trailing `\n`.
    ///
    /// # Examples
    ///
    /// ```
    /// let four_lines = "foo\nbar\n\nbaz";
    /// let v: Vec<&str> = four_lines.lines().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    ///
    /// Leaving off the trailing character:
    ///
    /// ```
    /// let four_lines = "foo\nbar\n\nbaz\n";
    /// let v: Vec<&str> = four_lines.lines().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lines(&self) -> Lines {
        core_str::StrExt::lines(&self[..])
    }

    /// An iterator over the lines of a string, separated by either
    /// `\n` or `\r\n`.
    ///
    /// As with `.lines()`, this does not include an empty trailing line.
    ///
    /// # Examples
    ///
    /// ```
    /// let four_lines = "foo\r\nbar\n\r\nbaz";
    /// let v: Vec<&str> = four_lines.lines_any().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    ///
    /// Leaving off the trailing character:
    ///
    /// ```
    /// let four_lines = "foo\r\nbar\n\r\nbaz\n";
    /// let v: Vec<&str> = four_lines.lines_any().collect();
    ///
    /// assert_eq!(v, ["foo", "bar", "", "baz"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn lines_any(&self) -> LinesAny {
        core_str::StrExt::lines_any(&self[..])
    }
    /// Returns a slice of the string from the character range [`begin`..`end`).
    ///
    /// That is, start at the `begin`-th code point of the string and continue
    /// to the `end`-th code point. This does not detect or handle edge cases
    /// such as leaving a combining character as the first code point of the
    /// string.
    ///
    /// Due to the design of UTF-8, this operation is `O(end)`. See `slice`,
    /// `slice_to` and `slice_from` for `O(1)` variants that use byte indices
    /// rather than code point indices.
    ///
    /// # Panics
    ///
    /// Panics if `begin` > `end` or the either `begin` or `end` are beyond the
    /// last character of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.slice_chars(0, 4), "Löwe");
    /// assert_eq!(s.slice_chars(5, 7), "老虎");
    /// ```
    #[unstable(feature = "collections",
               reason = "may have yet to prove its worth")]
    pub fn slice_chars(&self, begin: usize, end: usize) -> &str {
        core_str::StrExt::slice_chars(&self[..], begin, end)
    }

    /// Takes a bytewise slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// # Unsafety
    ///
    /// Caller must check both UTF-8 character boundaries and the boundaries
    /// of the entire slice as
    /// well.
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
    pub unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        core_str::StrExt::slice_unchecked(&self[..], begin, end)
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
        core_str::StrExt::starts_with(&self[..], pat)
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
        core_str::StrExt::ends_with(&self[..], pat)
    }

    /// Returns a string with all pre- and suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a simple `char`, or a closure that determines
    /// the split.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_matches(x), "foo1bar");
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// assert_eq!("123foo1bar123".trim_matches(|c: char| c.is_numeric()), "foo1bar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: DoubleEndedSearcher<'a>
    {
        core_str::StrExt::trim_matches(&self[..], pat)
    }

    /// Returns a string with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_left_matches(x), "foo1bar12");
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// assert_eq!("123foo1bar123".trim_left_matches(|c: char| c.is_numeric()), "foo1bar123");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_left_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str {
        core_str::StrExt::trim_left_matches(&self[..], pat)
    }

    /// Returns a string with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the split.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_right_matches(x), "12foo1bar");
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// assert_eq!("123foo1bar123".trim_right_matches(|c: char| c.is_numeric()), "123foo1bar");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_right_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>
    {
        core_str::StrExt::trim_right_matches(&self[..], pat)
    }

    /// Check that `index`-th byte lies at the start and/or end of a
    /// UTF-8 code point sequence.
    ///
    /// The start and end of the string (when `index == self.len()`) are
    /// considered to be
    /// boundaries.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(str_char)]
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
                         slicing")]
    pub fn is_char_boundary(&self, index: usize) -> bool {
        core_str::StrExt::is_char_boundary(&self[..], index)
    }

    /// Given a byte position, return the next char and its index.
    ///
    /// This can be used to iterate over the Unicode characters of a string.
    ///
    /// # Panics
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 character.
    ///
    /// # Examples
    ///
    /// This example manually iterates through the characters of a string;
    /// this should normally be
    /// done by `.chars()` or `.char_indices()`.
    ///
    /// ```
    /// # #![feature(str_char, core)]
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
    #[unstable(feature = "str_char",
               reason = "often replaced by char_indices, this method may \
                         be removed in favor of just char_at() or eventually \
                         removed altogether")]
    pub fn char_range_at(&self, start: usize) -> CharRange {
        core_str::StrExt::char_range_at(&self[..], start)
    }

    /// Given a byte position, return the previous `char` and its position.
    ///
    /// This function can be used to iterate over a Unicode string in reverse.
    ///
    /// Returns 0 for next index if called on start index 0.
    ///
    /// # Panics
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 character.
    ///
    /// # Examples
    ///
    /// This example manually iterates through the characters of a string;
    /// this should normally be
    /// done by `.chars().rev()` or `.char_indices()`.
    ///
    /// ```
    /// # #![feature(str_char, core)]
    /// use std::str::CharRange;
    ///
    /// let s = "中华Việt Nam";
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
    /// 16: m
    /// 15: a
    /// 14: N
    /// 13:
    /// 12: t
    /// 11: ệ
    /// 8: i
    /// 7: V
    /// 6: 华
    /// 3: 中
    /// ```
    #[unstable(feature = "str_char",
               reason = "often replaced by char_indices, this method may \
                         be removed in favor of just char_at_reverse() or \
                         eventually removed altogether")]
    pub fn char_range_at_reverse(&self, start: usize) -> CharRange {
        core_str::StrExt::char_range_at_reverse(&self[..], start)
    }

    /// Given a byte position, return the `char` at that position.
    ///
    /// # Panics
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 character.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(str_char)]
    /// let s = "abπc";
    /// assert_eq!(s.char_at(1), 'b');
    /// assert_eq!(s.char_at(2), 'π');
    /// ```
    #[unstable(feature = "str_char",
               reason = "frequently replaced by the chars() iterator, this \
                         method may be removed or possibly renamed in the \
                         future; it is normally replaced by chars/char_indices \
                         iterators or by getting the first char from a \
                         subslice")]
    pub fn char_at(&self, i: usize) -> char {
        core_str::StrExt::char_at(&self[..], i)
    }

    /// Given a byte position, return the `char` at that position, counting
    /// from the end.
    ///
    /// # Panics
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 character.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(str_char)]
    /// let s = "abπc";
    /// assert_eq!(s.char_at_reverse(1), 'a');
    /// assert_eq!(s.char_at_reverse(2), 'b');
    /// ```
    #[unstable(feature = "str_char",
               reason = "see char_at for more details, but reverse semantics \
                         are also somewhat unclear, especially with which \
                         cases generate panics")]
    pub fn char_at_reverse(&self, i: usize) -> char {
        core_str::StrExt::char_at_reverse(&self[..], i)
    }

    /// Convert `self` to a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!("bors".as_bytes(), b"bors");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_bytes(&self) -> &[u8] {
        core_str::StrExt::as_bytes(&self[..])
    }

    /// Returns the byte index of the first character of `self` that matches
    /// the pattern, if it
    /// exists.
    ///
    /// Returns `None` if it doesn't exist.
    ///
    /// The pattern can be a simple `&str`, `char`, or a closure that
    /// determines the
    /// split.
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
    /// assert_eq!(s.find(|c: char| c.is_whitespace()), Some(5));
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
        core_str::StrExt::find(&self[..], pat)
    }

    /// Returns the byte index of the last character of `self` that
    /// matches the pattern, if it
    /// exists.
    ///
    /// Returns `None` if it doesn't exist.
    ///
    /// The pattern can be a simple `&str`, `char`,
    /// or a closure that determines the split.
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
    /// assert_eq!(s.rfind(|c: char| c.is_whitespace()), Some(12));
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
        core_str::StrExt::rfind(&self[..], pat)
    }

    /// Retrieves the first character from a `&str` and returns it.
    ///
    /// This does not allocate a new string; instead, it returns a slice that
    /// points one character
    /// beyond the character that was shifted.
    ///
    /// If the slice does not contain any characters, None is returned instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(str_char)]
    /// let s = "Löwe 老虎 Léopard";
    /// let (c, s1) = s.slice_shift_char().unwrap();
    ///
    /// assert_eq!(c, 'L');
    /// assert_eq!(s1, "öwe 老虎 Léopard");
    ///
    /// let (c, s2) = s1.slice_shift_char().unwrap();
    ///
    /// assert_eq!(c, 'ö');
    /// assert_eq!(s2, "we 老虎 Léopard");
    /// ```
    #[unstable(feature = "str_char",
               reason = "awaiting conventions about shifting and slices and \
                         may not be warranted with the existence of the chars \
                         and/or char_indices iterators")]
    pub fn slice_shift_char(&self) -> Option<(char, &str)> {
        core_str::StrExt::slice_shift_char(&self[..])
    }

    /// Returns the byte offset of an inner slice relative to an enclosing
    /// outer slice.
    ///
    /// # Panics
    ///
    /// Panics if `inner` is not a direct slice contained within self.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(collections)]
    /// let string = "a\nb\nc";
    /// let lines: Vec<&str> = string.lines().collect();
    ///
    /// assert!(string.subslice_offset(lines[0]) == 0); // &"a"
    /// assert!(string.subslice_offset(lines[1]) == 2); // &"b"
    /// assert!(string.subslice_offset(lines[2]) == 4); // &"c"
    /// ```
    #[unstable(feature = "collections",
               reason = "awaiting convention about comparability of arbitrary slices")]
    pub fn subslice_offset(&self, inner: &str) -> usize {
        core_str::StrExt::subslice_offset(&self[..], inner)
    }

    /// Return an unsafe pointer to the `&str`'s buffer.
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
        core_str::StrExt::as_ptr(&self[..])
    }

    /// Return an iterator of `u16` over the string encoded as UTF-16.
    #[unstable(feature = "collections",
               reason = "this functionality may only be provided by libunicode")]
    pub fn utf16_units(&self) -> Utf16Units {
        Utf16Units { encoder: Utf16Encoder::new(self[..].chars()) }
    }

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
        core_str::StrExt::len(&self[..])
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
        core_str::StrExt::is_empty(&self[..])
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
        core_str::StrExt::parse(&self[..])
    }

    /// Returns an iterator over the [grapheme clusters][graphemes] of `self`.
    ///
    /// [graphemes]: http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries
    ///
    /// If `is_extended` is true, the iterator is over the
    /// *extended grapheme clusters*;
    /// otherwise, the iterator is over the *legacy grapheme clusters*.
    /// [UAX#29](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// recommends extended grapheme cluster boundaries for general processing.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(unicode, core)]
    /// let gr1 = "a\u{310}e\u{301}o\u{308}\u{332}".graphemes(true).collect::<Vec<&str>>();
    /// let b: &[_] = &["a\u{310}", "e\u{301}", "o\u{308}\u{332}"];
    ///
    /// assert_eq!(&gr1[..], b);
    ///
    /// let gr2 = "a\r\nb🇷🇺🇸🇹".graphemes(true).collect::<Vec<&str>>();
    /// let b: &[_] = &["a", "\r\n", "b", "🇷🇺🇸🇹"];
    ///
    /// assert_eq!(&gr2[..], b);
    /// ```
    #[unstable(feature = "unicode",
               reason = "this functionality may only be provided by libunicode")]
    pub fn graphemes(&self, is_extended: bool) -> Graphemes {
        UnicodeStr::graphemes(&self[..], is_extended)
    }

    /// Returns an iterator over the grapheme clusters of `self` and their
    /// byte offsets. See
    /// `graphemes()` for more information.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(unicode, core)]
    /// let gr_inds = "a̐éö̲\r\n".grapheme_indices(true).collect::<Vec<(usize, &str)>>();
    /// let b: &[_] = &[(0, "a̐"), (3, "é"), (6, "ö̲"), (11, "\r\n")];
    ///
    /// assert_eq!(&gr_inds[..], b);
    /// ```
    #[unstable(feature = "unicode",
               reason = "this functionality may only be provided by libunicode")]
    pub fn grapheme_indices(&self, is_extended: bool) -> GraphemeIndices {
        UnicodeStr::grapheme_indices(&self[..], is_extended)
    }

    /// An iterator over the non-empty words of `self`.
    ///
    /// A 'word' is a subsequence separated by any sequence of whitespace.
    /// Sequences of whitespace
    /// are collapsed, so empty "words" are not included.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(str_words)]
    /// let some_words = " Mary   had\ta little  \n\t lamb";
    /// let v: Vec<&str> = some_words.words().collect();
    ///
    /// assert_eq!(v, ["Mary", "had", "a", "little", "lamb"]);
    /// ```
    #[unstable(feature = "str_words",
               reason = "the precise algorithm to use is unclear")]
    pub fn words(&self) -> Words {
        UnicodeStr::words(&self[..])
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
        UnicodeStr::trim(&self[..])
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
        UnicodeStr::trim_left(&self[..])
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
        UnicodeStr::trim_right(&self[..])
    }

    /// Returns the lowercase equivalent of this string.
    ///
    /// # Examples
    ///
    /// let s = "HELLO";
    /// assert_eq!(s.to_lowercase(), "hello");
    #[unstable(feature = "collections")]
    pub fn to_lowercase(&self) -> String {
        let mut s = String::with_capacity(self.len());
        s.extend(self[..].chars().flat_map(|c| c.to_lowercase()));
        return s;
    }

    /// Returns the uppercase equivalent of this string.
    ///
    /// # Examples
    ///
    /// let s = "hello";
    /// assert_eq!(s.to_uppercase(), "HELLO");
    #[unstable(feature = "collections")]
    pub fn to_uppercase(&self) -> String {
        let mut s = String::with_capacity(self.len());
        s.extend(self[..].chars().flat_map(|c| c.to_uppercase()));
        return s;
    }
}
