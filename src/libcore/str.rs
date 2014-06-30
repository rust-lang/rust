// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! String manipulation
//!
//! For more details, see std::str

#![doc(primitive = "str")]

use mem;
use char;
use char::Char;
use clone::Clone;
use cmp;
use cmp::{PartialEq, Eq};
use collections::Collection;
use default::Default;
use iter::{Filter, Map, Iterator};
use iter::{DoubleEndedIterator, ExactSize};
use iter::range;
use num::{CheckedMul, Saturating};
use option::{None, Option, Some};
use raw::Repr;
use slice::ImmutableVector;
use slice;
use uint;

/*
Section: Creating a string
*/

/// Converts a vector to a string slice without performing any allocations.
///
/// Once the slice has been validated as utf-8, it is transmuted in-place and
/// returned as a '&str' instead of a '&[u8]'
///
/// Returns None if the slice is not utf-8.
pub fn from_utf8<'a>(v: &'a [u8]) -> Option<&'a str> {
    if is_utf8(v) {
        Some(unsafe { raw::from_utf8(v) })
    } else { None }
}

/// Something that can be used to compare against a character
pub trait CharEq {
    /// Determine if the splitter should split at the given character
    fn matches(&mut self, char) -> bool;
    /// Indicate if this is only concerned about ASCII characters,
    /// which can allow for a faster implementation.
    fn only_ascii(&self) -> bool;
}

impl CharEq for char {
    #[inline]
    fn matches(&mut self, c: char) -> bool { *self == c }

    #[inline]
    fn only_ascii(&self) -> bool { (*self as uint) < 128 }
}

impl<'a> CharEq for |char|: 'a -> bool {
    #[inline]
    fn matches(&mut self, c: char) -> bool { (*self)(c) }

    #[inline]
    fn only_ascii(&self) -> bool { false }
}

impl CharEq for extern "Rust" fn(char) -> bool {
    #[inline]
    fn matches(&mut self, c: char) -> bool { (*self)(c) }

    #[inline]
    fn only_ascii(&self) -> bool { false }
}

impl<'a> CharEq for &'a [char] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&mut m| m.matches(c))
    }

    #[inline]
    fn only_ascii(&self) -> bool {
        self.iter().all(|m| m.only_ascii())
    }
}

/*
Section: Iterators
*/

/// External iterator for a string's characters.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct Chars<'a> {
    /// The slice remaining to be iterated
    string: &'a str,
}

impl<'a> Iterator<char> for Chars<'a> {
    #[inline]
    fn next(&mut self) -> Option<char> {
        // Decode the next codepoint, then update
        // the slice to be just the remaining part
        if self.string.len() != 0 {
            let CharRange {ch, next} = self.string.char_range_at(0);
            unsafe {
                self.string = raw::slice_unchecked(self.string, next, self.string.len());
            }
            Some(ch)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.string.len().saturating_add(3)/4, Some(self.string.len()))
    }
}

impl<'a> DoubleEndedIterator<char> for Chars<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        if self.string.len() != 0 {
            let CharRange {ch, next} = self.string.char_range_at_reverse(self.string.len());
            unsafe {
                self.string = raw::slice_unchecked(self.string, 0, next);
            }
            Some(ch)
        } else {
            None
        }
    }
}

/// External iterator for a string's characters and their byte offsets.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct CharOffsets<'a> {
    /// The original string to be iterated
    string: &'a str,
    iter: Chars<'a>,
}

impl<'a> Iterator<(uint, char)> for CharOffsets<'a> {
    #[inline]
    fn next(&mut self) -> Option<(uint, char)> {
        // Compute the byte offset by using the pointer offset between
        // the original string slice and the iterator's remaining part
        let offset = self.iter.string.as_ptr() as uint - self.string.as_ptr() as uint;
        self.iter.next().map(|ch| (offset, ch))
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator<(uint, char)> for CharOffsets<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, char)> {
        self.iter.next_back().map(|ch| {
            let offset = self.iter.string.len() +
                    self.iter.string.as_ptr() as uint - self.string.as_ptr() as uint;
            (offset, ch)
        })
    }
}

/// External iterator for a string's bytes.
/// Use with the `std::iter` module.
pub type Bytes<'a> =
    Map<'a, &'a u8, u8, slice::Items<'a, u8>>;

/// An iterator over the substrings of a string, separated by `sep`.
#[deriving(Clone)]
pub struct CharSplits<'a, Sep> {
    /// The slice remaining to be iterated
    string: &'a str,
    sep: Sep,
    /// Whether an empty string at the end is allowed
    allow_trailing_empty: bool,
    only_ascii: bool,
    finished: bool,
}

/// An iterator over the substrings of a string, separated by `sep`,
/// splitting at most `count` times.
#[deriving(Clone)]
pub struct CharSplitsN<'a, Sep> {
    iter: CharSplits<'a, Sep>,
    /// The number of splits remaining
    count: uint,
    invert: bool,
}

/// An iterator over the words of a string, separated by a sequence of whitespace
pub type Words<'a> =
    Filter<'a, &'a str, CharSplits<'a, extern "Rust" fn(char) -> bool>>;

/// An iterator over the lines of a string, separated by either `\n` or (`\r\n`).
pub type AnyLines<'a> =
    Map<'a, &'a str, &'a str, CharSplits<'a, char>>;

impl<'a, Sep> CharSplits<'a, Sep> {
    #[inline]
    fn get_end(&mut self) -> Option<&'a str> {
        if !self.finished && (self.allow_trailing_empty || self.string.len() > 0) {
            self.finished = true;
            Some(self.string)
        } else {
            None
        }
    }
}

impl<'a, Sep: CharEq> Iterator<&'a str> for CharSplits<'a, Sep> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.finished { return None }

        let mut next_split = None;
        if self.only_ascii {
            for (idx, byte) in self.string.bytes().enumerate() {
                if self.sep.matches(byte as char) && byte < 128u8 {
                    next_split = Some((idx, idx + 1));
                    break;
                }
            }
        } else {
            for (idx, ch) in self.string.char_indices() {
                if self.sep.matches(ch) {
                    next_split = Some((idx, self.string.char_range_at(idx).next));
                    break;
                }
            }
        }
        match next_split {
            Some((a, b)) => unsafe {
                let elt = raw::slice_unchecked(self.string, 0, a);
                self.string = raw::slice_unchecked(self.string, b, self.string.len());
                Some(elt)
            },
            None => self.get_end(),
        }
    }
}

impl<'a, Sep: CharEq> DoubleEndedIterator<&'a str>
for CharSplits<'a, Sep> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        if self.finished { return None }

        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.next_back() {
                Some(elt) if !elt.is_empty() => return Some(elt),
                _ => if self.finished { return None }
            }
        }
        let len = self.string.len();
        let mut next_split = None;

        if self.only_ascii {
            for (idx, byte) in self.string.bytes().enumerate().rev() {
                if self.sep.matches(byte as char) && byte < 128u8 {
                    next_split = Some((idx, idx + 1));
                    break;
                }
            }
        } else {
            for (idx, ch) in self.string.char_indices().rev() {
                if self.sep.matches(ch) {
                    next_split = Some((idx, self.string.char_range_at(idx).next));
                    break;
                }
            }
        }
        match next_split {
            Some((a, b)) => unsafe {
                let elt = raw::slice_unchecked(self.string, b, len);
                self.string = raw::slice_unchecked(self.string, 0, a);
                Some(elt)
            },
            None => { self.finished = true; Some(self.string) }
        }
    }
}

impl<'a, Sep: CharEq> Iterator<&'a str> for CharSplitsN<'a, Sep> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.count != 0 {
            self.count -= 1;
            if self.invert { self.iter.next_back() } else { self.iter.next() }
        } else {
            self.iter.get_end()
        }
    }
}

/// The internal state of an iterator that searches for matches of a substring
/// within a larger string using naive search
#[deriving(Clone)]
struct NaiveSearcher {
    position: uint
}

impl NaiveSearcher {
    fn new() -> NaiveSearcher {
        NaiveSearcher { position: 0 }
    }

    fn next(&mut self, haystack: &[u8], needle: &[u8]) -> Option<(uint, uint)> {
        while self.position + needle.len() <= haystack.len() {
            if haystack.slice(self.position, self.position + needle.len()) == needle {
                let matchPos = self.position;
                self.position += needle.len(); // add 1 for all matches
                return Some((matchPos, matchPos + needle.len()));
            } else {
                self.position += 1;
            }
        }
        None
    }
}

/// The internal state of an iterator that searches for matches of a substring
/// within a larger string using two-way search
#[deriving(Clone)]
struct TwoWaySearcher {
    // constants
    critPos: uint,
    period: uint,
    byteset: u64,

    // variables
    position: uint,
    memory: uint
}

impl TwoWaySearcher {
    fn new(needle: &[u8]) -> TwoWaySearcher {
        let (critPos1, period1) = TwoWaySearcher::maximal_suffix(needle, false);
        let (critPos2, period2) = TwoWaySearcher::maximal_suffix(needle, true);

        let critPos;
        let period;
        if critPos1 > critPos2 {
            critPos = critPos1;
            period = period1;
        } else {
            critPos = critPos2;
            period = period2;
        }

        let byteset = needle.iter()
                            .fold(0, |a, &b| (1 << ((b & 0x3f) as uint)) | a);

        if needle.slice_to(critPos) == needle.slice_from(needle.len() - critPos) {
            TwoWaySearcher {
                critPos: critPos,
                period: period,
                byteset: byteset,

                position: 0,
                memory: 0
            }
        } else {
            TwoWaySearcher {
                critPos: critPos,
                period: cmp::max(critPos, needle.len() - critPos) + 1,
                byteset: byteset,

                position: 0,
                memory: uint::MAX // Dummy value to signify that the period is long
            }
        }
    }

    #[inline]
    fn next(&mut self, haystack: &[u8], needle: &[u8], longPeriod: bool) -> Option<(uint, uint)> {
        'search: loop {
            // Check that we have room to search in
            if self.position + needle.len() > haystack.len() {
                return None;
            }

            // Quickly skip by large portions unrelated to our substring
            if (self.byteset >>
                    ((haystack[self.position + needle.len() - 1] & 0x3f)
                     as uint)) & 1 == 0 {
                self.position += needle.len();
                continue 'search;
            }

            // See if the right part of the needle matches
            let start = if longPeriod { self.critPos } else { cmp::max(self.critPos, self.memory) };
            for i in range(start, needle.len()) {
                if needle[i] != haystack[self.position + i] {
                    self.position += i - self.critPos + 1;
                    if !longPeriod {
                        self.memory = 0;
                    }
                    continue 'search;
                }
            }

            // See if the left part of the needle matches
            let start = if longPeriod { 0 } else { self.memory };
            for i in range(start, self.critPos).rev() {
                if needle[i] != haystack[self.position + i] {
                    self.position += self.period;
                    if !longPeriod {
                        self.memory = needle.len() - self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            let matchPos = self.position;
            self.position += needle.len(); // add self.period for all matches
            if !longPeriod {
                self.memory = 0; // set to needle.len() - self.period for all matches
            }
            return Some((matchPos, matchPos + needle.len()));
        }
    }

    #[inline]
    fn maximal_suffix(arr: &[u8], reversed: bool) -> (uint, uint) {
        let mut left = -1; // Corresponds to i in the paper
        let mut right = 0; // Corresponds to j in the paper
        let mut offset = 1; // Corresponds to k in the paper
        let mut period = 1; // Corresponds to p in the paper

        while right + offset < arr.len() {
            let a;
            let b;
            if reversed {
                a = arr[left + offset];
                b = arr[right + offset];
            } else {
                a = arr[right + offset];
                b = arr[left + offset];
            }
            if a < b {
                // Suffix is smaller, period is entire prefix so far.
                right += offset;
                offset = 1;
                period = right - left;
            } else if a == b {
                // Advance through repetition of the current period.
                if offset == period {
                    right += offset;
                    offset = 1;
                } else {
                    offset += 1;
                }
            } else {
                // Suffix is larger, start over from current location.
                left = right;
                right += 1;
                offset = 1;
                period = 1;
            }
        }
        (left + 1, period)
    }
}

/// The internal state of an iterator that searches for matches of a substring
/// within a larger string using a dynamically chosen search algorithm
#[deriving(Clone)]
enum Searcher {
    Naive(NaiveSearcher),
    TwoWay(TwoWaySearcher),
    TwoWayLong(TwoWaySearcher)
}

impl Searcher {
    fn new(haystack: &[u8], needle: &[u8]) -> Searcher {
        // FIXME: Tune this.
        if needle.len() > haystack.len() - 20 {
            Naive(NaiveSearcher::new())
        } else {
            let searcher = TwoWaySearcher::new(needle);
            if searcher.memory == uint::MAX { // If the period is long
                TwoWayLong(searcher)
            } else {
                TwoWay(searcher)
            }
        }
    }
}

/// An iterator over the start and end indices of the matches of a
/// substring within a larger string
#[deriving(Clone)]
pub struct MatchIndices<'a> {
    // constants
    haystack: &'a str,
    needle: &'a str,
    searcher: Searcher
}

/// An iterator over the substrings of a string separated by a given
/// search string
#[deriving(Clone)]
pub struct StrSplits<'a> {
    it: MatchIndices<'a>,
    last_end: uint,
    finished: bool
}

impl<'a> Iterator<(uint, uint)> for MatchIndices<'a> {
    #[inline]
    fn next(&mut self) -> Option<(uint, uint)> {
        match self.searcher {
            Naive(ref mut searcher)
                => searcher.next(self.haystack.as_bytes(), self.needle.as_bytes()),
            TwoWay(ref mut searcher)
                => searcher.next(self.haystack.as_bytes(), self.needle.as_bytes(), false),
            TwoWayLong(ref mut searcher)
                => searcher.next(self.haystack.as_bytes(), self.needle.as_bytes(), true)
        }
    }
}

impl<'a> Iterator<&'a str> for StrSplits<'a> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.finished { return None; }

        match self.it.next() {
            Some((from, to)) => {
                let ret = Some(self.it.haystack.slice(self.last_end, from));
                self.last_end = to;
                ret
            }
            None => {
                self.finished = true;
                Some(self.it.haystack.slice(self.last_end, self.it.haystack.len()))
            }
        }
    }
}

/// External iterator for a string's UTF16 codeunits.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct Utf16CodeUnits<'a> {
    chars: Chars<'a>,
    extra: u16
}

impl<'a> Iterator<u16> for Utf16CodeUnits<'a> {
    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0u16, ..2];
        self.chars.next().map(|ch| {
            let n = ch.encode_utf16(buf /* as mut slice! */);
            if n == 2 { self.extra = buf[1]; }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (low, high) = self.chars.size_hint();
        // every char gets either one u16 or two u16,
        // so this iterator is between 1 or 2 times as
        // long as the underlying iterator.
        (low, high.and_then(|n| n.checked_mul(&2)))
    }
}

/*
Section: Comparing strings
*/

// share the implementation of the lang-item vs. non-lang-item
// eq_slice.
/// NOTE: This function is (ab)used in rustc::middle::trans::_match
/// to compare &[u8] byte slices that are not necessarily valid UTF-8.
#[inline]
fn eq_slice_(a: &str, b: &str) -> bool {
    #[allow(ctypes)]
    extern { fn memcmp(s1: *const i8, s2: *const i8, n: uint) -> i32; }
    a.len() == b.len() && unsafe {
        memcmp(a.as_ptr() as *const i8,
               b.as_ptr() as *const i8,
               a.len()) == 0
    }
}

/// Bytewise slice equality
/// NOTE: This function is (ab)used in rustc::middle::trans::_match
/// to compare &[u8] byte slices that are not necessarily valid UTF-8.
#[lang="str_eq"]
#[inline]
pub fn eq_slice(a: &str, b: &str) -> bool {
    eq_slice_(a, b)
}

/*
Section: Misc
*/

/// Walk through `iter` checking that it's a valid UTF-8 sequence,
/// returning `true` in that case, or, if it is invalid, `false` with
/// `iter` reset such that it is pointing at the first byte in the
/// invalid sequence.
#[inline(always)]
fn run_utf8_validation_iterator(iter: &mut slice::Items<u8>) -> bool {
    loop {
        // save the current thing we're pointing at.
        let old = *iter;

        // restore the iterator we had at the start of this codepoint.
        macro_rules! err ( () => { {*iter = old; return false} });
        macro_rules! next ( () => {
                match iter.next() {
                    Some(a) => *a,
                    // we needed data, but there was none: error!
                    None => err!()
                }
            });

        let first = match iter.next() {
            Some(&b) => b,
            // we're at the end of the iterator and a codepoint
            // boundary at the same time, so this string is valid.
            None => return true
        };

        // ASCII characters are always valid, so only large
        // bytes need more examination.
        if first >= 128 {
            let w = utf8_char_width(first);
            let second = next!();
            // 2-byte encoding is for codepoints  \u0080 to  \u07ff
            //        first  C2 80        last DF BF
            // 3-byte encoding is for codepoints  \u0800 to  \uffff
            //        first  E0 A0 80     last EF BF BF
            //   excluding surrogates codepoints  \ud800 to  \udfff
            //               ED A0 80 to       ED BF BF
            // 4-byte encoding is for codepoints \u10000 to \u10ffff
            //        first  F0 90 80 80  last F4 8F BF BF
            //
            // Use the UTF-8 syntax from the RFC
            //
            // https://tools.ietf.org/html/rfc3629
            // UTF8-1      = %x00-7F
            // UTF8-2      = %xC2-DF UTF8-tail
            // UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
            //               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
            // UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
            //               %xF4 %x80-8F 2( UTF8-tail )
            match w {
                2 => if second & 192 != TAG_CONT_U8 {err!()},
                3 => {
                    match (first, second, next!() & 192) {
                        (0xE0        , 0xA0 .. 0xBF, TAG_CONT_U8) |
                        (0xE1 .. 0xEC, 0x80 .. 0xBF, TAG_CONT_U8) |
                        (0xED        , 0x80 .. 0x9F, TAG_CONT_U8) |
                        (0xEE .. 0xEF, 0x80 .. 0xBF, TAG_CONT_U8) => {}
                        _ => err!()
                    }
                }
                4 => {
                    match (first, second, next!() & 192, next!() & 192) {
                        (0xF0        , 0x90 .. 0xBF, TAG_CONT_U8, TAG_CONT_U8) |
                        (0xF1 .. 0xF3, 0x80 .. 0xBF, TAG_CONT_U8, TAG_CONT_U8) |
                        (0xF4        , 0x80 .. 0x8F, TAG_CONT_U8, TAG_CONT_U8) => {}
                        _ => err!()
                    }
                }
                _ => err!()
            }
        }
    }
}

/// Determines if a vector of bytes contains valid UTF-8.
pub fn is_utf8(v: &[u8]) -> bool {
    run_utf8_validation_iterator(&mut v.iter())
}

/// Determines if a vector of `u16` contains valid UTF-16
pub fn is_utf16(v: &[u16]) -> bool {
    let mut it = v.iter();
    macro_rules! next ( ($ret:expr) => {
            match it.next() { Some(u) => *u, None => return $ret }
        }
    )
    loop {
        let u = next!(true);

        match char::from_u32(u as u32) {
            Some(_) => {}
            None => {
                let u2 = next!(false);
                if u < 0xD7FF || u > 0xDBFF ||
                    u2 < 0xDC00 || u2 > 0xDFFF { return false; }
            }
        }
    }
}

/// An iterator that decodes UTF-16 encoded codepoints from a vector
/// of `u16`s.
#[deriving(Clone)]
pub struct Utf16Items<'a> {
    iter: slice::Items<'a, u16>
}
/// The possibilities for values decoded from a `u16` stream.
#[deriving(PartialEq, Eq, Clone, Show)]
pub enum Utf16Item {
    /// A valid codepoint.
    ScalarValue(char),
    /// An invalid surrogate without its pair.
    LoneSurrogate(u16)
}

impl Utf16Item {
    /// Convert `self` to a `char`, taking `LoneSurrogate`s to the
    /// replacement character (U+FFFD).
    #[inline]
    pub fn to_char_lossy(&self) -> char {
        match *self {
            ScalarValue(c) => c,
            LoneSurrogate(_) => '\uFFFD'
        }
    }
}

impl<'a> Iterator<Utf16Item> for Utf16Items<'a> {
    fn next(&mut self) -> Option<Utf16Item> {
        let u = match self.iter.next() {
            Some(u) => *u,
            None => return None
        };

        if u < 0xD800 || 0xDFFF < u {
            // not a surrogate
            Some(ScalarValue(unsafe {mem::transmute(u as u32)}))
        } else if u >= 0xDC00 {
            // a trailing surrogate
            Some(LoneSurrogate(u))
        } else {
            // preserve state for rewinding.
            let old = self.iter;

            let u2 = match self.iter.next() {
                Some(u2) => *u2,
                // eof
                None => return Some(LoneSurrogate(u))
            };
            if u2 < 0xDC00 || u2 > 0xDFFF {
                // not a trailing surrogate so we're not a valid
                // surrogate pair, so rewind to redecode u2 next time.
                self.iter = old;
                return Some(LoneSurrogate(u))
            }

            // all ok, so lets decode it.
            let c = ((u - 0xD800) as u32 << 10 | (u2 - 0xDC00) as u32) + 0x1_0000;
            Some(ScalarValue(unsafe {mem::transmute(c)}))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (low, high) = self.iter.size_hint();
        // we could be entirely valid surrogates (2 elements per
        // char), or entirely non-surrogates (1 element per char)
        (low / 2, high)
    }
}

/// Create an iterator over the UTF-16 encoded codepoints in `v`,
/// returning invalid surrogates as `LoneSurrogate`s.
///
/// # Example
///
/// ```rust
/// use std::str;
/// use std::str::{ScalarValue, LoneSurrogate};
///
/// // 𝄞mus<invalid>ic<invalid>
/// let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///          0x0073, 0xDD1E, 0x0069, 0x0063,
///          0xD834];
///
/// assert_eq!(str::utf16_items(v).collect::<Vec<_>>(),
///            vec![ScalarValue('𝄞'),
///                 ScalarValue('m'), ScalarValue('u'), ScalarValue('s'),
///                 LoneSurrogate(0xDD1E),
///                 ScalarValue('i'), ScalarValue('c'),
///                 LoneSurrogate(0xD834)]);
/// ```
pub fn utf16_items<'a>(v: &'a [u16]) -> Utf16Items<'a> {
    Utf16Items { iter : v.iter() }
}

/// Return a slice of `v` ending at (and not including) the first NUL
/// (0).
///
/// # Example
///
/// ```rust
/// use std::str;
///
/// // "abcd"
/// let mut v = ['a' as u16, 'b' as u16, 'c' as u16, 'd' as u16];
/// // no NULs so no change
/// assert_eq!(str::truncate_utf16_at_nul(v), v.as_slice());
///
/// // "ab\0d"
/// v[2] = 0;
/// assert_eq!(str::truncate_utf16_at_nul(v),
///            &['a' as u16, 'b' as u16]);
/// ```
pub fn truncate_utf16_at_nul<'a>(v: &'a [u16]) -> &'a [u16] {
    match v.iter().position(|c| *c == 0) {
        // don't include the 0
        Some(i) => v.slice_to(i),
        None => v
    }
}

// https://tools.ietf.org/html/rfc3629
static UTF8_CHAR_WIDTH: [u8, ..256] = [
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
pub fn utf8_char_width(b: u8) -> uint {
    return UTF8_CHAR_WIDTH[b as uint] as uint;
}

/// Struct that contains a `char` and the index of the first byte of
/// the next `char` in a string.  This can be used as a data structure
/// for iterating over the UTF-8 bytes of a string.
pub struct CharRange {
    /// Current `char`
    pub ch: char,
    /// Index of the first byte of the next `char`
    pub next: uint,
}

// Return the initial codepoint accumulator for the first byte.
// The first byte is special, only want bottom 5 bits for width 2, 4 bits
// for width 3, and 3 bits for width 4
macro_rules! utf8_first_byte(
    ($byte:expr, $width:expr) => (($byte & (0x7F >> $width)) as u32)
)

// return the value of $ch updated with continuation byte $byte
macro_rules! utf8_acc_cont_byte(
    ($ch:expr, $byte:expr) => (($ch << 6) | ($byte & 63u8) as u32)
)

static TAG_CONT_U8: u8 = 128u8;

/// Unsafe operations
pub mod raw {
    use mem;
    use collections::Collection;
    use ptr::RawPtr;
    use raw::Slice;
    use slice::{ImmutableVector};
    use str::{is_utf8, StrSlice};

    /// Converts a slice of bytes to a string slice without checking
    /// that the string contains valid UTF-8.
    pub unsafe fn from_utf8<'a>(v: &'a [u8]) -> &'a str {
        mem::transmute(v)
    }

    /// Form a slice from a C string. Unsafe because the caller must ensure the
    /// C string has the static lifetime, or else the return value may be
    /// invalidated later.
    pub unsafe fn c_str_to_static_slice(s: *const i8) -> &'static str {
        let s = s as *const u8;
        let mut curr = s;
        let mut len = 0u;
        while *curr != 0u8 {
            len += 1u;
            curr = s.offset(len as int);
        }
        let v = Slice { data: s, len: len };
        assert!(is_utf8(::mem::transmute(v)));
        ::mem::transmute(v)
    }

    /// Takes a bytewise (not UTF-8) slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// # Failure
    ///
    /// If begin is greater than end.
    /// If end is greater than the length of the string.
    #[inline]
    pub unsafe fn slice_bytes<'a>(s: &'a str, begin: uint, end: uint) -> &'a str {
        assert!(begin <= end);
        assert!(end <= s.len());
        slice_unchecked(s, begin, end)
    }

    /// Takes a bytewise (not UTF-8) slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// Caller must check slice boundaries!
    #[inline]
    pub unsafe fn slice_unchecked<'a>(s: &'a str, begin: uint, end: uint) -> &'a str {
        mem::transmute(Slice {
                data: s.as_ptr().offset(begin as int),
                len: end - begin,
            })
    }
}

/*
Section: Trait implementations
*/

#[allow(missing_doc)]
pub mod traits {
    use cmp::{Ord, Ordering, Less, Equal, Greater, PartialEq, PartialOrd, Equiv, Eq};
    use collections::Collection;
    use iter::Iterator;
    use option::{Option, Some, None};
    use str::{Str, StrSlice, eq_slice};

    impl<'a> Ord for &'a str {
        #[inline]
        fn cmp(&self, other: & &'a str) -> Ordering {
            for (s_b, o_b) in self.bytes().zip(other.bytes()) {
                match s_b.cmp(&o_b) {
                    Greater => return Greater,
                    Less => return Less,
                    Equal => ()
                }
            }

            self.len().cmp(&other.len())
        }
    }

    impl<'a> PartialEq for &'a str {
        #[inline]
        fn eq(&self, other: & &'a str) -> bool {
            eq_slice((*self), (*other))
        }
        #[inline]
        fn ne(&self, other: & &'a str) -> bool { !(*self).eq(other) }
    }

    impl<'a> Eq for &'a str {}

    impl<'a> PartialOrd for &'a str {
        #[inline]
        fn partial_cmp(&self, other: &&'a str) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<'a, S: Str> Equiv<S> for &'a str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(*self, other.as_slice()) }
    }
}

/// Any string that can be represented as a slice
pub trait Str {
    /// Work with `self` as a slice.
    fn as_slice<'a>(&'a self) -> &'a str;
}

impl<'a> Str for &'a str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str { *self }
}

impl<'a> Collection for &'a str {
    #[inline]
    fn len(&self) -> uint {
        self.repr().len
    }
}

/// Methods for string slices
pub trait StrSlice<'a> {
    /// Returns true if one string contains another
    ///
    /// # Arguments
    ///
    /// - needle - The string to look for
    fn contains<'a>(&self, needle: &'a str) -> bool;

    /// Returns true if a string contains a char.
    ///
    /// # Arguments
    ///
    /// - needle - The char to look for
    fn contains_char(&self, needle: char) -> bool;

    /// An iterator over the characters of `self`. Note, this iterates
    /// over unicode code-points, not unicode graphemes.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<char> = "abc åäö".chars().collect();
    /// assert_eq!(v, vec!['a', 'b', 'c', ' ', 'å', 'ä', 'ö']);
    /// ```
    fn chars(&self) -> Chars<'a>;

    /// An iterator over the bytes of `self`
    fn bytes(&self) -> Bytes<'a>;

    /// An iterator over the characters of `self` and their byte offsets.
    fn char_indices(&self) -> CharOffsets<'a>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "Mary had a little lamb".split(' ').collect();
    /// assert_eq!(v, vec!["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".split(|c: char| c.is_digit()).collect();
    /// assert_eq!(v, vec!["abc", "def", "ghi"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').collect();
    /// assert_eq!(v, vec!["lion", "", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "".split('X').collect();
    /// assert_eq!(v, vec![""]);
    /// ```
    fn split<Sep: CharEq>(&self, sep: Sep) -> CharSplits<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, restricted to splitting at most `count`
    /// times.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "Mary had a little lambda".splitn(' ', 2).collect();
    /// assert_eq!(v, vec!["Mary", "had", "a little lambda"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".splitn(|c: char| c.is_digit(), 1).collect();
    /// assert_eq!(v, vec!["abc", "def2ghi"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".splitn('X', 2).collect();
    /// assert_eq!(v, vec!["lion", "", "tigerXleopard"]);
    ///
    /// let v: Vec<&str> = "abcXdef".splitn('X', 0).collect();
    /// assert_eq!(v, vec!["abcXdef"]);
    ///
    /// let v: Vec<&str> = "".splitn('X', 1).collect();
    /// assert_eq!(v, vec![""]);
    /// ```
    fn splitn<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitsN<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`.
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
    /// let v: Vec<&str> = "abc1def2ghi".split(|c: char| c.is_digit()).rev().collect();
    /// assert_eq!(v, vec!["ghi", "def", "abc"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').rev().collect();
    /// assert_eq!(v, vec!["leopard", "tiger", "", "lion"]);
    /// ```
    fn split_terminator<Sep: CharEq>(&self, sep: Sep) -> CharSplits<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, starting from the end of the string.
    /// Restricted to splitting at most `count` times.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<&str> = "Mary had a little lamb".rsplitn(' ', 2).collect();
    /// assert_eq!(v, vec!["lamb", "little", "Mary had a"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".rsplitn(|c: char| c.is_digit(), 1).collect();
    /// assert_eq!(v, vec!["ghi", "abc1def"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplitn('X', 2).collect();
    /// assert_eq!(v, vec!["leopard", "tiger", "lionX"]);
    /// ```
    fn rsplitn<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitsN<'a, Sep>;

    /// An iterator over the start and end indices of the disjoint
    /// matches of `sep` within `self`.
    ///
    /// That is, each returned value `(start, end)` satisfies
    /// `self.slice(start, end) == sep`. For matches of `sep` within
    /// `self` that overlap, only the indices corresponding to the
    /// first match are returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: Vec<(uint, uint)> = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, vec![(0,3), (6,9), (12,15)]);
    ///
    /// let v: Vec<(uint, uint)> = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, vec![(1,4), (4,7)]);
    ///
    /// let v: Vec<(uint, uint)> = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, vec![(0, 3)]); // only the first `aba`
    /// ```
    fn match_indices(&self, sep: &'a str) -> MatchIndices<'a>;

    /// An iterator over the substrings of `self` separated by `sep`.
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
    fn split_str(&self, &'a str) -> StrSplits<'a>;

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
    fn lines(&self) -> CharSplits<'a, char>;

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
    fn lines_any(&self) -> AnyLines<'a>;

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
    fn words(&self) -> Words<'a>;

    /// Returns true if the string contains only whitespace.
    ///
    /// Whitespace characters are determined by `char::is_whitespace`.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!(" \t\n".is_whitespace());
    /// assert!("".is_whitespace());
    ///
    /// assert!( !"abc".is_whitespace());
    /// ```
    fn is_whitespace(&self) -> bool;

    /// Returns true if the string contains only alphanumeric code
    /// points.
    ///
    /// Alphanumeric characters are determined by `char::is_alphanumeric`.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!("Löwe老虎Léopard123".is_alphanumeric());
    /// assert!("".is_alphanumeric());
    ///
    /// assert!( !" &*~".is_alphanumeric());
    /// ```
    fn is_alphanumeric(&self) -> bool;

    /// Returns the number of Unicode code points (`char`) that a
    /// string holds.
    ///
    /// This does not perform any normalization, and is `O(n)`, since
    /// UTF-8 is a variable width encoding of code points.
    ///
    /// *Warning*: The number of code points in a string does not directly
    /// correspond to the number of visible characters or width of the
    /// visible text due to composing characters, and double- and
    /// zero-width ones.
    ///
    /// See also `.len()` for the byte length.
    ///
    /// # Example
    ///
    /// ```rust
    /// // composed forms of `ö` and `é`
    /// let c = "Löwe 老虎 Léopard"; // German, Simplified Chinese, French
    /// // decomposed forms of `ö` and `é`
    /// let d = "Lo\u0308we 老虎 Le\u0301opard";
    ///
    /// assert_eq!(c.char_len(), 15);
    /// assert_eq!(d.char_len(), 17);
    ///
    /// assert_eq!(c.len(), 21);
    /// assert_eq!(d.len(), 23);
    ///
    /// // the two strings *look* the same
    /// println!("{}", c);
    /// println!("{}", d);
    /// ```
    fn char_len(&self) -> uint;

    /// Returns a slice of the given string from the byte range
    /// [`begin`..`end`).
    ///
    /// This operation is `O(1)`.
    ///
    /// Fails when `begin` and `end` do not point to valid characters
    /// or point beyond the last character of the string.
    ///
    /// See also `slice_to` and `slice_from` for slicing prefixes and
    /// suffixes of strings, and `slice_chars` for slicing based on
    /// code point counts.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// assert_eq!(s.slice(0, 1), "L");
    ///
    /// assert_eq!(s.slice(1, 9), "öwe 老");
    ///
    /// // these will fail:
    /// // byte 2 lies within `ö`:
    /// // s.slice(2, 3);
    ///
    /// // byte 8 lies within `老`
    /// // s.slice(1, 8);
    ///
    /// // byte 100 is outside the string
    /// // s.slice(3, 100);
    /// ```
    fn slice(&self, begin: uint, end: uint) -> &'a str;

    /// Returns a slice of the string from `begin` to its end.
    ///
    /// Equivalent to `self.slice(begin, self.len())`.
    ///
    /// Fails when `begin` does not point to a valid character, or is
    /// out of bounds.
    ///
    /// See also `slice`, `slice_to` and `slice_chars`.
    fn slice_from(&self, begin: uint) -> &'a str;

    /// Returns a slice of the string from the beginning to byte
    /// `end`.
    ///
    /// Equivalent to `self.slice(0, end)`.
    ///
    /// Fails when `end` does not point to a valid character, or is
    /// out of bounds.
    ///
    /// See also `slice`, `slice_from` and `slice_chars`.
    fn slice_to(&self, end: uint) -> &'a str;

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
    /// Fails if `begin` > `end` or the either `begin` or `end` are
    /// beyond the last character of the string.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// assert_eq!(s.slice_chars(0, 4), "Löwe");
    /// assert_eq!(s.slice_chars(5, 7), "老虎");
    /// ```
    fn slice_chars(&self, begin: uint, end: uint) -> &'a str;

    /// Returns true if `needle` is a prefix of the string.
    fn starts_with(&self, needle: &str) -> bool;

    /// Returns true if `needle` is a suffix of the string.
    fn ends_with(&self, needle: &str) -> bool;

    /// Returns a string with leading and trailing whitespace removed.
    fn trim(&self) -> &'a str;

    /// Returns a string with leading whitespace removed.
    fn trim_left(&self) -> &'a str;

    /// Returns a string with trailing whitespace removed.
    fn trim_right(&self) -> &'a str;

    /// Returns a string with characters that match `to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_chars('1'), "foo1bar")
    /// assert_eq!("12foo1bar12".trim_chars(&['1', '2']), "foo1bar")
    /// assert_eq!("123foo1bar123".trim_chars(|c: char| c.is_digit()), "foo1bar")
    /// ```
    fn trim_chars<C: CharEq>(&self, to_trim: C) -> &'a str;

    /// Returns a string with leading `chars_to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_left_chars('1'), "foo1bar11")
    /// assert_eq!("12foo1bar12".trim_left_chars(&['1', '2']), "foo1bar12")
    /// assert_eq!("123foo1bar123".trim_left_chars(|c: char| c.is_digit()), "foo1bar123")
    /// ```
    fn trim_left_chars<C: CharEq>(&self, to_trim: C) -> &'a str;

    /// Returns a string with trailing `chars_to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_right_chars('1'), "11foo1bar")
    /// assert_eq!("12foo1bar12".trim_right_chars(&['1', '2']), "12foo1bar")
    /// assert_eq!("123foo1bar123".trim_right_chars(|c: char| c.is_digit()), "123foo1bar")
    /// ```
    fn trim_right_chars<C: CharEq>(&self, to_trim: C) -> &'a str;

    /// Check that `index`-th byte lies at the start and/or end of a
    /// UTF-8 code point sequence.
    ///
    /// The start and end of the string (when `index == self.len()`)
    /// are considered to be boundaries.
    ///
    /// Fails if `index` is greater than `self.len()`.
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
    fn is_char_boundary(&self, index: uint) -> bool;

    /// Pluck a character out of a string and return the index of the next
    /// character.
    ///
    /// This function can be used to iterate over the unicode characters of a
    /// string.
    ///
    /// # Example
    ///
    /// This example manually iterate through the characters of a
    /// string; this should normally by done by `.chars()` or
    /// `.char_indices`.
    ///
    /// ```rust
    /// use std::str::CharRange;
    ///
    /// let s = "中华Việt Nam";
    /// let mut i = 0u;
    /// while i < s.len() {
    ///     let CharRange {ch, next} = s.char_range_at(i);
    ///     println!("{}: {}", i, ch);
    ///     i = next;
    /// }
    /// ```
    ///
    /// ## Output
    ///
    /// ```ignore
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
    /// A record {ch: char, next: uint} containing the char value and the byte
    /// index of the next unicode character.
    ///
    /// # Failure
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 character.
    fn char_range_at(&self, start: uint) -> CharRange;

    /// Given a byte position and a str, return the previous char and its position.
    ///
    /// This function can be used to iterate over a unicode string in reverse.
    ///
    /// Returns 0 for next index if called on start index 0.
    ///
    /// # Failure
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 character.
    fn char_range_at_reverse(&self, start: uint) -> CharRange;

    /// Plucks the character starting at the `i`th byte of a string.
    ///
    /// # Failure
    ///
    /// If `i` is greater than or equal to the length of the string.
    /// If `i` is not the index of the beginning of a valid UTF-8 character.
    fn char_at(&self, i: uint) -> char;

    /// Plucks the character ending at the `i`th byte of a string.
    ///
    /// # Failure
    ///
    /// If `i` is greater than the length of the string.
    /// If `i` is not an index following a valid UTF-8 character.
    fn char_at_reverse(&self, i: uint) -> char;

    /// Work with the byte buffer of a string as a byte slice.
    fn as_bytes(&self) -> &'a [u8];

    /// Returns the byte index of the first character of `self` that
    /// matches `search`.
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
    /// assert_eq!(s.find(&['1', '2']), None);
    /// ```
    fn find<C: CharEq>(&self, search: C) -> Option<uint>;

    /// Returns the byte index of the last character of `self` that
    /// matches `search`.
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
    /// assert_eq!(s.rfind(&['1', '2']), None);
    /// ```
    fn rfind<C: CharEq>(&self, search: C) -> Option<uint>;

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
    fn find_str(&self, &str) -> Option<uint>;

    /// Retrieves the first character from a string slice and returns
    /// it. This does not allocate a new string; instead, it returns a
    /// slice that point one character beyond the character that was
    /// shifted. If the string does not contain any characters,
    /// a tuple of None and an empty string is returned instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// let (c, s1) = s.slice_shift_char();
    /// assert_eq!(c, Some('L'));
    /// assert_eq!(s1, "öwe 老虎 Léopard");
    ///
    /// let (c, s2) = s1.slice_shift_char();
    /// assert_eq!(c, Some('ö'));
    /// assert_eq!(s2, "we 老虎 Léopard");
    /// ```
    fn slice_shift_char(&self) -> (Option<char>, &'a str);

    /// Returns the byte offset of an inner slice relative to an enclosing outer slice.
    ///
    /// Fails if `inner` is not a direct slice contained within self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let string = "a\nb\nc";
    /// let lines: Vec<&str> = string.lines().collect();
    /// let lines = lines.as_slice();
    ///
    /// assert!(string.subslice_offset(lines[0]) == 0); // &"a"
    /// assert!(string.subslice_offset(lines[1]) == 2); // &"b"
    /// assert!(string.subslice_offset(lines[2]) == 4); // &"c"
    /// ```
    fn subslice_offset(&self, inner: &str) -> uint;

    /// Return an unsafe pointer to the strings buffer.
    ///
    /// The caller must ensure that the string outlives this pointer,
    /// and that it is not reallocated (e.g. by pushing to the
    /// string).
    fn as_ptr(&self) -> *const u8;

    /// Return an iterator of `u16` over the string encoded as UTF-16.
    fn utf16_units(&self) -> Utf16CodeUnits<'a>;
}

impl<'a> StrSlice<'a> for &'a str {
    #[inline]
    fn contains<'a>(&self, needle: &'a str) -> bool {
        self.find_str(needle).is_some()
    }

    #[inline]
    fn contains_char(&self, needle: char) -> bool {
        self.find(needle).is_some()
    }

    #[inline]
    fn chars(&self) -> Chars<'a> {
        Chars{string: *self}
    }

    #[inline]
    fn bytes(&self) -> Bytes<'a> {
        self.as_bytes().iter().map(|&b| b)
    }

    #[inline]
    fn char_indices(&self) -> CharOffsets<'a> {
        CharOffsets{string: *self, iter: self.chars()}
    }

    #[inline]
    fn split<Sep: CharEq>(&self, sep: Sep) -> CharSplits<'a, Sep> {
        CharSplits {
            string: *self,
            only_ascii: sep.only_ascii(),
            sep: sep,
            allow_trailing_empty: true,
            finished: false,
        }
    }

    #[inline]
    fn splitn<Sep: CharEq>(&self, sep: Sep, count: uint)
        -> CharSplitsN<'a, Sep> {
        CharSplitsN {
            iter: self.split(sep),
            count: count,
            invert: false,
        }
    }

    #[inline]
    fn split_terminator<Sep: CharEq>(&self, sep: Sep)
        -> CharSplits<'a, Sep> {
        CharSplits {
            allow_trailing_empty: false,
            ..self.split(sep)
        }
    }

    #[inline]
    fn rsplitn<Sep: CharEq>(&self, sep: Sep, count: uint)
        -> CharSplitsN<'a, Sep> {
        CharSplitsN {
            iter: self.split(sep),
            count: count,
            invert: true,
        }
    }

    #[inline]
    fn match_indices(&self, sep: &'a str) -> MatchIndices<'a> {
        assert!(!sep.is_empty())
        MatchIndices {
            haystack: *self,
            needle: sep,
            searcher: Searcher::new(self.as_bytes(), sep.as_bytes())
        }
    }

    #[inline]
    fn split_str(&self, sep: &'a str) -> StrSplits<'a> {
        StrSplits {
            it: self.match_indices(sep),
            last_end: 0,
            finished: false
        }
    }

    #[inline]
    fn lines(&self) -> CharSplits<'a, char> {
        self.split_terminator('\n')
    }

    fn lines_any(&self) -> AnyLines<'a> {
        self.lines().map(|line| {
            let l = line.len();
            if l > 0 && line[l - 1] == '\r' as u8 { line.slice(0, l - 1) }
            else { line }
        })
    }

    #[inline]
    fn words(&self) -> Words<'a> {
        self.split(char::is_whitespace).filter(|s| !s.is_empty())
    }

    #[inline]
    fn is_whitespace(&self) -> bool { self.chars().all(char::is_whitespace) }

    #[inline]
    fn is_alphanumeric(&self) -> bool { self.chars().all(char::is_alphanumeric) }

    #[inline]
    fn char_len(&self) -> uint { self.chars().count() }

    #[inline]
    fn slice(&self, begin: uint, end: uint) -> &'a str {
        assert!(self.is_char_boundary(begin) && self.is_char_boundary(end));
        unsafe { raw::slice_bytes(*self, begin, end) }
    }

    #[inline]
    fn slice_from(&self, begin: uint) -> &'a str {
        self.slice(begin, self.len())
    }

    #[inline]
    fn slice_to(&self, end: uint) -> &'a str {
        assert!(self.is_char_boundary(end));
        unsafe { raw::slice_bytes(*self, 0, end) }
    }

    fn slice_chars(&self, begin: uint, end: uint) -> &'a str {
        assert!(begin <= end);
        let mut count = 0;
        let mut begin_byte = None;
        let mut end_byte = None;

        // This could be even more efficient by not decoding,
        // only finding the char boundaries
        for (idx, _) in self.char_indices() {
            if count == begin { begin_byte = Some(idx); }
            if count == end { end_byte = Some(idx); break; }
            count += 1;
        }
        if begin_byte.is_none() && count == begin { begin_byte = Some(self.len()) }
        if end_byte.is_none() && count == end { end_byte = Some(self.len()) }

        match (begin_byte, end_byte) {
            (None, _) => fail!("slice_chars: `begin` is beyond end of string"),
            (_, None) => fail!("slice_chars: `end` is beyond end of string"),
            (Some(a), Some(b)) => unsafe { raw::slice_bytes(*self, a, b) }
        }
    }

    #[inline]
    fn starts_with<'a>(&self, needle: &'a str) -> bool {
        let n = needle.len();
        self.len() >= n && needle.as_bytes() == self.as_bytes().slice_to(n)
    }

    #[inline]
    fn ends_with(&self, needle: &str) -> bool {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle.as_bytes() == self.as_bytes().slice_from(m - n)
    }

    #[inline]
    fn trim(&self) -> &'a str {
        self.trim_left().trim_right()
    }

    #[inline]
    fn trim_left(&self) -> &'a str {
        self.trim_left_chars(char::is_whitespace)
    }

    #[inline]
    fn trim_right(&self) -> &'a str {
        self.trim_right_chars(char::is_whitespace)
    }

    #[inline]
    fn trim_chars<C: CharEq>(&self, mut to_trim: C) -> &'a str {
        let cur = match self.find(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(i) => unsafe { raw::slice_bytes(*self, i, self.len()) }
        };
        match cur.rfind(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(i) => {
                let right = cur.char_range_at(i).next;
                unsafe { raw::slice_bytes(cur, 0, right) }
            }
        }
    }

    #[inline]
    fn trim_left_chars<C: CharEq>(&self, mut to_trim: C) -> &'a str {
        match self.find(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(first) => unsafe { raw::slice_bytes(*self, first, self.len()) }
        }
    }

    #[inline]
    fn trim_right_chars<C: CharEq>(&self, mut to_trim: C) -> &'a str {
        match self.rfind(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(last) => {
                let next = self.char_range_at(last).next;
                unsafe { raw::slice_bytes(*self, 0u, next) }
            }
        }
    }

    #[inline]
    fn is_char_boundary(&self, index: uint) -> bool {
        if index == self.len() { return true; }
        if index > self.len() { return false; }
        let b = self[index];
        return b < 128u8 || b >= 192u8;
    }

    #[inline]
    fn char_range_at(&self, i: uint) -> CharRange {
        if self[i] < 128u8 {
            return CharRange {ch: self[i] as char, next: i + 1 };
        }

        // Multibyte case is a fn to allow char_range_at to inline cleanly
        fn multibyte_char_range_at(s: &str, i: uint) -> CharRange {
            let mut val = s[i] as u32;
            let w = UTF8_CHAR_WIDTH[val as uint] as uint;
            assert!((w != 0));

            val = utf8_first_byte!(val, w);
            val = utf8_acc_cont_byte!(val, s[i + 1]);
            if w > 2 { val = utf8_acc_cont_byte!(val, s[i + 2]); }
            if w > 3 { val = utf8_acc_cont_byte!(val, s[i + 3]); }

            return CharRange {ch: unsafe { mem::transmute(val) }, next: i + w};
        }

        return multibyte_char_range_at(*self, i);
    }

    #[inline]
    fn char_range_at_reverse(&self, start: uint) -> CharRange {
        let mut prev = start;

        prev = prev.saturating_sub(1);
        if self[prev] < 128 { return CharRange{ch: self[prev] as char, next: prev} }

        // Multibyte case is a fn to allow char_range_at_reverse to inline cleanly
        fn multibyte_char_range_at_reverse(s: &str, mut i: uint) -> CharRange {
            // while there is a previous byte == 10......
            while i > 0 && s[i] & 192u8 == TAG_CONT_U8 {
                i -= 1u;
            }

            let mut val = s[i] as u32;
            let w = UTF8_CHAR_WIDTH[val as uint] as uint;
            assert!((w != 0));

            val = utf8_first_byte!(val, w);
            val = utf8_acc_cont_byte!(val, s[i + 1]);
            if w > 2 { val = utf8_acc_cont_byte!(val, s[i + 2]); }
            if w > 3 { val = utf8_acc_cont_byte!(val, s[i + 3]); }

            return CharRange {ch: unsafe { mem::transmute(val) }, next: i};
        }

        return multibyte_char_range_at_reverse(*self, prev);
    }

    #[inline]
    fn char_at(&self, i: uint) -> char {
        self.char_range_at(i).ch
    }

    #[inline]
    fn char_at_reverse(&self, i: uint) -> char {
        self.char_range_at_reverse(i).ch
    }

    #[inline]
    fn as_bytes(&self) -> &'a [u8] {
        unsafe { mem::transmute(*self) }
    }

    fn find<C: CharEq>(&self, mut search: C) -> Option<uint> {
        if search.only_ascii() {
            self.bytes().position(|b| search.matches(b as char))
        } else {
            for (index, c) in self.char_indices() {
                if search.matches(c) { return Some(index); }
            }
            None
        }
    }

    fn rfind<C: CharEq>(&self, mut search: C) -> Option<uint> {
        if search.only_ascii() {
            self.bytes().rposition(|b| search.matches(b as char))
        } else {
            for (index, c) in self.char_indices().rev() {
                if search.matches(c) { return Some(index); }
            }
            None
        }
    }

    fn find_str(&self, needle: &str) -> Option<uint> {
        if needle.is_empty() {
            Some(0)
        } else {
            self.match_indices(needle)
                .next()
                .map(|(start, _end)| start)
        }
    }

    #[inline]
    fn slice_shift_char(&self) -> (Option<char>, &'a str) {
        if self.is_empty() {
            return (None, *self);
        } else {
            let CharRange {ch, next} = self.char_range_at(0u);
            let next_s = unsafe { raw::slice_bytes(*self, next, self.len()) };
            return (Some(ch), next_s);
        }
    }

    fn subslice_offset(&self, inner: &str) -> uint {
        let a_start = self.as_ptr() as uint;
        let a_end = a_start + self.len();
        let b_start = inner.as_ptr() as uint;
        let b_end = b_start + inner.len();

        assert!(a_start <= b_start);
        assert!(b_end <= a_end);
        b_start - a_start
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.repr().data
    }

    #[inline]
    fn utf16_units(&self) -> Utf16CodeUnits<'a> {
        Utf16CodeUnits{ chars: self.chars(), extra: 0}
    }
}

impl<'a> Default for &'a str {
    fn default() -> &'a str { "" }
}
