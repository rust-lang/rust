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

//! String manipulation
//!
//! For more details, see std::str

#![doc(primitive = "str")]

use self::Searcher::{Naive, TwoWay, TwoWayLong};

use clone::Clone;
use cmp::{mod, Eq};
use default::Default;
use iter::range;
use iter::{DoubleEndedIteratorExt, ExactSizeIterator};
use iter::{Map, Iterator, IteratorExt, DoubleEndedIterator};
use kinds::Sized;
use mem;
use num::Int;
use ops::{Fn, FnMut};
use option::Option::{mod, None, Some};
use ptr::RawPtr;
use raw::{Repr, Slice};
use result::Result::{mod, Ok, Err};
use slice::{mod, SliceExt};
use uint;

/// A trait to abstract the idea of creating a new instance of a type from a
/// string.
// FIXME(#17307): there should be an `E` associated type for a `Result` return
#[unstable = "will return a Result once associated types are working"]
pub trait FromStr {
    /// Parses a string `s` to return an optional value of this type. If the
    /// string is ill-formatted, the None is returned.
    fn from_str(s: &str) -> Option<Self>;
}

/// A utility function that just calls FromStr::from_str
#[deprecated = "call the .parse() method on the string instead"]
pub fn from_str<A: FromStr>(s: &str) -> Option<A> {
    FromStr::from_str(s)
}

impl FromStr for bool {
    /// Parse a `bool` from a string.
    ///
    /// Yields an `Option<bool>`, because `s` may or may not actually be parseable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!("true".parse(), Some(true));
    /// assert_eq!("false".parse(), Some(false));
    /// assert_eq!("not even a boolean".parse::<bool>(), None);
    /// ```
    #[inline]
    fn from_str(s: &str) -> Option<bool> {
        match s {
            "true"  => Some(true),
            "false" => Some(false),
            _       => None,
        }
    }
}

/*
Section: Creating a string
*/

/// Errors which can occur when attempting to interpret a byte slice as a `str`.
#[deriving(Copy, Eq, PartialEq, Clone)]
pub enum Utf8Error {
    /// An invalid byte was detected at the byte offset given.
    ///
    /// The offset is guaranteed to be in bounds of the slice in question, and
    /// the byte at the specified offset was the first invalid byte in the
    /// sequence detected.
    InvalidByte(uint),

    /// The byte slice was invalid because more bytes were needed but no more
    /// bytes were available.
    TooShort,
}

/// Converts a slice of bytes to a string slice without performing any
/// allocations.
///
/// Once the slice has been validated as utf-8, it is transmuted in-place and
/// returned as a '&str' instead of a '&[u8]'
///
/// # Failure
///
/// Returns `Err` if the slice is not utf-8 with a description as to why the
/// provided slice is not utf-8.
pub fn from_utf8(v: &[u8]) -> Result<&str, Utf8Error> {
    try!(run_utf8_validation_iterator(&mut v.iter()));
    Ok(unsafe { from_utf8_unchecked(v) })
}

/// Converts a slice of bytes to a string slice without checking
/// that the string contains valid UTF-8.
#[stable]
pub unsafe fn from_utf8_unchecked<'a>(v: &'a [u8]) -> &'a str {
    mem::transmute(v)
}

/// Constructs a static string slice from a given raw pointer.
///
/// This function will read memory starting at `s` until it finds a 0, and then
/// transmute the memory up to that point as a string slice, returning the
/// corresponding `&'static str` value.
///
/// This function is unsafe because the caller must ensure the C string itself
/// has the static lifetime and that the memory `s` is valid up to and including
/// the first null byte.
///
/// # Panics
///
/// This function will panic if the string pointed to by `s` is not valid UTF-8.
#[unstable = "may change location based on the outcome of the c_str module"]
pub unsafe fn from_c_str(s: *const i8) -> &'static str {
    let s = s as *const u8;
    let mut len = 0u;
    while *s.offset(len as int) != 0 {
        len += 1u;
    }
    let v: &'static [u8] = ::mem::transmute(Slice { data: s, len: len });
    from_utf8(v).ok().expect("from_c_str passed invalid utf-8 data")
}

/// Something that can be used to compare against a character
#[unstable = "definition may change as pattern-related methods are stabilized"]
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

impl<F> CharEq for F where F: FnMut(char) -> bool {
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

/// Iterator for the char (representing *Unicode Scalar Values*) of a string
///
/// Created with the method `.chars()`.
#[deriving(Clone, Copy)]
pub struct Chars<'a> {
    iter: slice::Iter<'a, u8>
}

// Return the initial codepoint accumulator for the first byte.
// The first byte is special, only want bottom 5 bits for width 2, 4 bits
// for width 3, and 3 bits for width 4
macro_rules! utf8_first_byte {
    ($byte:expr, $width:expr) => (($byte & (0x7F >> $width)) as u32)
}

// return the value of $ch updated with continuation byte $byte
macro_rules! utf8_acc_cont_byte {
    ($ch:expr, $byte:expr) => (($ch << 6) | ($byte & CONT_MASK) as u32)
}

macro_rules! utf8_is_cont_byte {
    ($byte:expr) => (($byte & !CONT_MASK) == TAG_CONT_U8)
}

#[inline]
fn unwrap_or_0(opt: Option<&u8>) -> u8 {
    match opt {
        Some(&byte) => byte,
        None => 0,
    }
}

impl<'a> Iterator<char> for Chars<'a> {
    #[inline]
    fn next(&mut self) -> Option<char> {
        // Decode UTF-8, using the valid UTF-8 invariant
        let x = match self.iter.next() {
            None => return None,
            Some(&next_byte) if next_byte < 128 => return Some(next_byte as char),
            Some(&next_byte) => next_byte,
        };

        // Multibyte case follows
        // Decode from a byte combination out of: [[[x y] z] w]
        // NOTE: Performance is sensitive to the exact formulation here
        let init = utf8_first_byte!(x, 2);
        let y = unwrap_or_0(self.iter.next());
        let mut ch = utf8_acc_cont_byte!(init, y);
        if x >= 0xE0 {
            // [[x y z] w] case
            // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
            let z = unwrap_or_0(self.iter.next());
            let y_z = utf8_acc_cont_byte!((y & CONT_MASK) as u32, z);
            ch = init << 12 | y_z;
            if x >= 0xF0 {
                // [x y z w] case
                // use only the lower 3 bits of `init`
                let w = unwrap_or_0(self.iter.next());
                ch = (init & 7) << 18 | utf8_acc_cont_byte!(y_z, w);
            }
        }

        // str invariant says `ch` is a valid Unicode Scalar Value
        unsafe {
            Some(mem::transmute(ch))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (len, _) = self.iter.size_hint();
        (len.saturating_add(3) / 4, Some(len))
    }
}

impl<'a> DoubleEndedIterator<char> for Chars<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        let w = match self.iter.next_back() {
            None => return None,
            Some(&back_byte) if back_byte < 128 => return Some(back_byte as char),
            Some(&back_byte) => back_byte,
        };

        // Multibyte case follows
        // Decode from a byte combination out of: [x [y [z w]]]
        let mut ch;
        let z = unwrap_or_0(self.iter.next_back());
        ch = utf8_first_byte!(z, 2);
        if utf8_is_cont_byte!(z) {
            let y = unwrap_or_0(self.iter.next_back());
            ch = utf8_first_byte!(y, 3);
            if utf8_is_cont_byte!(y) {
                let x = unwrap_or_0(self.iter.next_back());
                ch = utf8_first_byte!(x, 4);
                ch = utf8_acc_cont_byte!(ch, y);
            }
            ch = utf8_acc_cont_byte!(ch, z);
        }
        ch = utf8_acc_cont_byte!(ch, w);

        // str invariant says `ch` is a valid Unicode Scalar Value
        unsafe {
            Some(mem::transmute(ch))
        }
    }
}

/// External iterator for a string's characters and their byte offsets.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct CharIndices<'a> {
    front_offset: uint,
    iter: Chars<'a>,
}

impl<'a> Iterator<(uint, char)> for CharIndices<'a> {
    #[inline]
    fn next(&mut self) -> Option<(uint, char)> {
        let (pre_len, _) = self.iter.iter.size_hint();
        match self.iter.next() {
            None => None,
            Some(ch) => {
                let index = self.front_offset;
                let (len, _) = self.iter.iter.size_hint();
                self.front_offset += pre_len - len;
                Some((index, ch))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator<(uint, char)> for CharIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, char)> {
        match self.iter.next_back() {
            None => None,
            Some(ch) => {
                let (len, _) = self.iter.iter.size_hint();
                let index = self.front_offset + len;
                Some((index, ch))
            }
        }
    }
}

/// External iterator for a string's bytes.
/// Use with the `std::iter` module.
#[stable]
#[deriving(Clone)]
pub struct Bytes<'a> {
    inner: Map<&'a u8, u8, slice::Iter<'a, u8>, BytesFn>,
}

/// A temporary new type wrapper that ensures that the `Bytes` iterator
/// is cloneable.
#[deriving(Copy)]
struct BytesFn(fn(&u8) -> u8);

impl<'a> Fn(&'a u8) -> u8 for BytesFn {
    extern "rust-call" fn call(&self, (ptr,): (&'a u8,)) -> u8 {
        (self.0)(ptr)
    }
}

impl Clone for BytesFn {
    fn clone(&self) -> BytesFn { *self }
}

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

/// An iterator over the lines of a string, separated by `\n`.
#[stable]
pub struct Lines<'a> {
    inner: CharSplits<'a, char>,
}

/// An iterator over the lines of a string, separated by either `\n` or (`\r\n`).
#[stable]
pub struct LinesAny<'a> {
    inner: Map<&'a str, &'a str, Lines<'a>, fn(&str) -> &str>,
}

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
                let elt = self.string.slice_unchecked(0, a);
                self.string = self.string.slice_unchecked(b, self.string.len());
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
                let elt = self.string.slice_unchecked(b, len);
                self.string = self.string.slice_unchecked(0, a);
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
            if haystack[self.position .. self.position + needle.len()] == needle {
                let match_pos = self.position;
                self.position += needle.len(); // add 1 for all matches
                return Some((match_pos, match_pos + needle.len()));
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
    crit_pos: uint,
    period: uint,
    byteset: u64,

    // variables
    position: uint,
    memory: uint
}

/*
    This is the Two-Way search algorithm, which was introduced in the paper:
    Crochemore, M., Perrin, D., 1991, Two-way string-matching, Journal of the ACM 38(3):651-675.

    Here's some background information.

    A *word* is a string of symbols. The *length* of a word should be a familiar
    notion, and here we denote it for any word x by |x|.
    (We also allow for the possibility of the *empty word*, a word of length zero).

    If x is any non-empty word, then an integer p with 0 < p <= |x| is said to be a
    *period* for x iff for all i with 0 <= i <= |x| - p - 1, we have x[i] == x[i+p].
    For example, both 1 and 2 are periods for the string "aa". As another example,
    the only period of the string "abcd" is 4.

    We denote by period(x) the *smallest* period of x (provided that x is non-empty).
    This is always well-defined since every non-empty word x has at least one period,
    |x|. We sometimes call this *the period* of x.

    If u, v and x are words such that x = uv, where uv is the concatenation of u and
    v, then we say that (u, v) is a *factorization* of x.

    Let (u, v) be a factorization for a word x. Then if w is a non-empty word such
    that both of the following hold

      - either w is a suffix of u or u is a suffix of w
      - either w is a prefix of v or v is a prefix of w

    then w is said to be a *repetition* for the factorization (u, v).

    Just to unpack this, there are four possibilities here. Let w = "abc". Then we
    might have:

      - w is a suffix of u and w is a prefix of v. ex: ("lolabc", "abcde")
      - w is a suffix of u and v is a prefix of w. ex: ("lolabc", "ab")
      - u is a suffix of w and w is a prefix of v. ex: ("bc", "abchi")
      - u is a suffix of w and v is a prefix of w. ex: ("bc", "a")

    Note that the word vu is a repetition for any factorization (u,v) of x = uv,
    so every factorization has at least one repetition.

    If x is a string and (u, v) is a factorization for x, then a *local period* for
    (u, v) is an integer r such that there is some word w such that |w| = r and w is
    a repetition for (u, v).

    We denote by local_period(u, v) the smallest local period of (u, v). We sometimes
    call this *the local period* of (u, v). Provided that x = uv is non-empty, this
    is well-defined (because each non-empty word has at least one factorization, as
    noted above).

    It can be proven that the following is an equivalent definition of a local period
    for a factorization (u, v): any positive integer r such that x[i] == x[i+r] for
    all i such that |u| - r <= i <= |u| - 1 and such that both x[i] and x[i+r] are
    defined. (i.e. i > 0 and i + r < |x|).

    Using the above reformulation, it is easy to prove that

        1 <= local_period(u, v) <= period(uv)

    A factorization (u, v) of x such that local_period(u,v) = period(x) is called a
    *critical factorization*.

    The algorithm hinges on the following theorem, which is stated without proof:

    **Critical Factorization Theorem** Any word x has at least one critical
    factorization (u, v) such that |u| < period(x).

    The purpose of maximal_suffix is to find such a critical factorization.

*/
impl TwoWaySearcher {
    fn new(needle: &[u8]) -> TwoWaySearcher {
        let (crit_pos1, period1) = TwoWaySearcher::maximal_suffix(needle, false);
        let (crit_pos2, period2) = TwoWaySearcher::maximal_suffix(needle, true);

        let crit_pos;
        let period;
        if crit_pos1 > crit_pos2 {
            crit_pos = crit_pos1;
            period = period1;
        } else {
            crit_pos = crit_pos2;
            period = period2;
        }

        // This isn't in the original algorithm, as far as I'm aware.
        let byteset = needle.iter()
                            .fold(0, |a, &b| (1 << ((b & 0x3f) as uint)) | a);

        // A particularly readable explanation of what's going on here can be found
        // in Crochemore and Rytter's book "Text Algorithms", ch 13. Specifically
        // see the code for "Algorithm CP" on p. 323.
        //
        // What's going on is we have some critical factorization (u, v) of the
        // needle, and we want to determine whether u is a suffix of
        // v[..period]. If it is, we use "Algorithm CP1". Otherwise we use
        // "Algorithm CP2", which is optimized for when the period of the needle
        // is large.
        if needle[..crit_pos] == needle[period.. period + crit_pos] {
            TwoWaySearcher {
                crit_pos: crit_pos,
                period: period,
                byteset: byteset,

                position: 0,
                memory: 0
            }
        } else {
            TwoWaySearcher {
                crit_pos: crit_pos,
                period: cmp::max(crit_pos, needle.len() - crit_pos) + 1,
                byteset: byteset,

                position: 0,
                memory: uint::MAX // Dummy value to signify that the period is long
            }
        }
    }

    // One of the main ideas of Two-Way is that we factorize the needle into
    // two halves, (u, v), and begin trying to find v in the haystack by scanning
    // left to right. If v matches, we try to match u by scanning right to left.
    // How far we can jump when we encounter a mismatch is all based on the fact
    // that (u, v) is a critical factorization for the needle.
    #[inline]
    fn next(&mut self, haystack: &[u8], needle: &[u8], long_period: bool) -> Option<(uint, uint)> {
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
                if !long_period {
                    self.memory = 0;
                }
                continue 'search;
            }

            // See if the right part of the needle matches
            let start = if long_period { self.crit_pos }
                        else { cmp::max(self.crit_pos, self.memory) };
            for i in range(start, needle.len()) {
                if needle[i] != haystack[self.position + i] {
                    self.position += i - self.crit_pos + 1;
                    if !long_period {
                        self.memory = 0;
                    }
                    continue 'search;
                }
            }

            // See if the left part of the needle matches
            let start = if long_period { 0 } else { self.memory };
            for i in range(start, self.crit_pos).rev() {
                if needle[i] != haystack[self.position + i] {
                    self.position += self.period;
                    if !long_period {
                        self.memory = needle.len() - self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            let match_pos = self.position;
            self.position += needle.len(); // add self.period for all matches
            if !long_period {
                self.memory = 0; // set to needle.len() - self.period for all matches
            }
            return Some((match_pos, match_pos + needle.len()));
        }
    }

    // Computes a critical factorization (u, v) of `arr`.
    // Specifically, returns (i, p), where i is the starting index of v in some
    // critical factorization (u, v) and p = period(v)
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
        // FIXME(#16715): This unsigned integer addition will probably not
        // overflow because that would mean that the memory almost solely
        // consists of the needle. Needs #16715 to be formally fixed.
        if needle.len() + 20 > haystack.len() {
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

/*
Section: Comparing strings
*/

// share the implementation of the lang-item vs. non-lang-item
// eq_slice.
/// NOTE: This function is (ab)used in rustc::middle::trans::_match
/// to compare &[u8] byte slices that are not necessarily valid UTF-8.
#[inline]
fn eq_slice_(a: &str, b: &str) -> bool {
    #[allow(improper_ctypes)]
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
fn eq_slice(a: &str, b: &str) -> bool {
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
fn run_utf8_validation_iterator(iter: &mut slice::Iter<u8>)
                                -> Result<(), Utf8Error> {
    let whole = iter.as_slice();
    loop {
        // save the current thing we're pointing at.
        let old = *iter;

        // restore the iterator we had at the start of this codepoint.
        macro_rules! err (() => { {
            *iter = old;
            return Err(Utf8Error::InvalidByte(whole.len() - iter.as_slice().len()))
        } });
        macro_rules! next ( () => {
            match iter.next() {
                Some(a) => *a,
                // we needed data, but there was none: error!
                None => return Err(Utf8Error::TooShort),
            }
        });

        let first = match iter.next() {
            Some(&b) => b,
            // we're at the end of the iterator and a codepoint
            // boundary at the same time, so this string is valid.
            None => return Ok(())
        };

        // ASCII characters are always valid, so only large
        // bytes need more examination.
        if first >= 128 {
            let w = UTF8_CHAR_WIDTH[first as uint] as uint;
            let second = next!();
            // 2-byte encoding is for codepoints  \u{0080} to  \u{07ff}
            //        first  C2 80        last DF BF
            // 3-byte encoding is for codepoints  \u{0800} to  \u{ffff}
            //        first  E0 A0 80     last EF BF BF
            //   excluding surrogates codepoints  \u{d800} to  \u{dfff}
            //               ED A0 80 to       ED BF BF
            // 4-byte encoding is for codepoints \u{1000}0 to \u{10ff}ff
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
                2 => if second & !CONT_MASK != TAG_CONT_U8 {err!()},
                3 => {
                    match (first, second, next!() & !CONT_MASK) {
                        (0xE0         , 0xA0 ... 0xBF, TAG_CONT_U8) |
                        (0xE1 ... 0xEC, 0x80 ... 0xBF, TAG_CONT_U8) |
                        (0xED         , 0x80 ... 0x9F, TAG_CONT_U8) |
                        (0xEE ... 0xEF, 0x80 ... 0xBF, TAG_CONT_U8) => {}
                        _ => err!()
                    }
                }
                4 => {
                    match (first, second, next!() & !CONT_MASK, next!() & !CONT_MASK) {
                        (0xF0         , 0x90 ... 0xBF, TAG_CONT_U8, TAG_CONT_U8) |
                        (0xF1 ... 0xF3, 0x80 ... 0xBF, TAG_CONT_U8, TAG_CONT_U8) |
                        (0xF4         , 0x80 ... 0x8F, TAG_CONT_U8, TAG_CONT_U8) => {}
                        _ => err!()
                    }
                }
                _ => err!()
            }
        }
    }
}

/// Determines if a vector of bytes contains valid UTF-8.
#[deprecated = "call from_utf8 instead"]
pub fn is_utf8(v: &[u8]) -> bool {
    run_utf8_validation_iterator(&mut v.iter()).is_ok()
}

/// Deprecated function
#[deprecated = "this function will be removed"]
pub fn truncate_utf16_at_nul<'a>(v: &'a [u16]) -> &'a [u16] {
    match v.iter().position(|c| *c == 0) {
        // don't include the 0
        Some(i) => v[..i],
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
#[deprecated = "this function has moved to libunicode"]
pub fn utf8_char_width(b: u8) -> uint {
    return UTF8_CHAR_WIDTH[b as uint] as uint;
}

/// Struct that contains a `char` and the index of the first byte of
/// the next `char` in a string.  This can be used as a data structure
/// for iterating over the UTF-8 bytes of a string.
#[deriving(Copy)]
#[unstable = "naming is uncertain with container conventions"]
pub struct CharRange {
    /// Current `char`
    pub ch: char,
    /// Index of the first byte of the next `char`
    pub next: uint,
}

/// Mask of the value bits of a continuation byte
const CONT_MASK: u8 = 0b0011_1111u8;
/// Value of the tag bits (tag mask is !CONT_MASK) of a continuation byte
const TAG_CONT_U8: u8 = 0b1000_0000u8;

/// Unsafe operations
#[deprecated]
pub mod raw {
    use ptr::RawPtr;
    use raw::Slice;
    use slice::SliceExt;
    use str::StrExt;

    /// Converts a slice of bytes to a string slice without checking
    /// that the string contains valid UTF-8.
    #[deprecated = "renamed to str::from_utf8_unchecked"]
    pub unsafe fn from_utf8<'a>(v: &'a [u8]) -> &'a str {
        super::from_utf8_unchecked(v)
    }

    /// Form a slice from a C string. Unsafe because the caller must ensure the
    /// C string has the static lifetime, or else the return value may be
    /// invalidated later.
    #[deprecated = "renamed to str::from_c_str"]
    pub unsafe fn c_str_to_static_slice(s: *const i8) -> &'static str {
        let s = s as *const u8;
        let mut curr = s;
        let mut len = 0u;
        while *curr != 0u8 {
            len += 1u;
            curr = s.offset(len as int);
        }
        let v = Slice { data: s, len: len };
        super::from_utf8(::mem::transmute(v)).unwrap()
    }

    /// Takes a bytewise (not UTF-8) slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// # Panics
    ///
    /// If begin is greater than end.
    /// If end is greater than the length of the string.
    #[inline]
    #[deprecated = "call the slice_unchecked method instead"]
    pub unsafe fn slice_bytes<'a>(s: &'a str, begin: uint, end: uint) -> &'a str {
        assert!(begin <= end);
        assert!(end <= s.len());
        s.slice_unchecked(begin, end)
    }

    /// Takes a bytewise (not UTF-8) slice from a string.
    ///
    /// Returns the substring from [`begin`..`end`).
    ///
    /// Caller must check slice boundaries!
    #[inline]
    #[deprecated = "this has moved to a method on `str` directly"]
    pub unsafe fn slice_unchecked<'a>(s: &'a str, begin: uint, end: uint) -> &'a str {
        s.slice_unchecked(begin, end)
    }
}

/*
Section: Trait implementations
*/

#[allow(missing_docs)]
pub mod traits {
    use cmp::{Ordering, Ord, PartialEq, PartialOrd, Equiv, Eq};
    use cmp::Ordering::{Less, Equal, Greater};
    use iter::IteratorExt;
    use option::Option;
    use option::Option::Some;
    use ops;
    use str::{Str, StrExt, eq_slice};

    impl Ord for str {
        #[inline]
        fn cmp(&self, other: &str) -> Ordering {
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

    impl PartialEq for str {
        #[inline]
        fn eq(&self, other: &str) -> bool {
            eq_slice(self, other)
        }
        #[inline]
        fn ne(&self, other: &str) -> bool { !(*self).eq(other) }
    }

    impl Eq for str {}

    impl PartialOrd for str {
        #[inline]
        fn partial_cmp(&self, other: &str) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[allow(deprecated)]
    #[deprecated = "Use overloaded `core::cmp::PartialEq`"]
    impl<S: Str> Equiv<S> for str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(self, other.as_slice()) }
    }

    impl ops::Slice<uint, str> for str {
        #[inline]
        fn as_slice_<'a>(&'a self) -> &'a str {
            self
        }

        #[inline]
        fn slice_from_or_fail<'a>(&'a self, from: &uint) -> &'a str {
            self.slice_from(*from)
        }

        #[inline]
        fn slice_to_or_fail<'a>(&'a self, to: &uint) -> &'a str {
            self.slice_to(*to)
        }

        #[inline]
        fn slice_or_fail<'a>(&'a self, from: &uint, to: &uint) -> &'a str {
            self.slice(*from, *to)
        }
    }
}

/// Any string that can be represented as a slice
#[unstable = "Instead of taking this bound generically, this trait will be \
              replaced with one of slicing syntax, deref coercions, or \
              a more generic conversion trait"]
pub trait Str for Sized? {
    /// Work with `self` as a slice.
    fn as_slice<'a>(&'a self) -> &'a str;
}

#[allow(deprecated)]
impl Str for str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str { self }
}

#[allow(deprecated)]
impl<'a, Sized? S> Str for &'a S where S: Str {
    #[inline]
    fn as_slice(&self) -> &str { Str::as_slice(*self) }
}

/// Methods for string slices
#[allow(missing_docs)]
pub trait StrExt for Sized? {
    // NB there are no docs here are they're all located on the StrExt trait in
    // libcollections, not here.

    fn contains(&self, needle: &str) -> bool;
    fn contains_char(&self, needle: char) -> bool;
    fn chars<'a>(&'a self) -> Chars<'a>;
    fn bytes<'a>(&'a self) -> Bytes<'a>;
    fn char_indices<'a>(&'a self) -> CharIndices<'a>;
    fn split<'a, Sep: CharEq>(&'a self, sep: Sep) -> CharSplits<'a, Sep>;
    fn splitn<'a, Sep: CharEq>(&'a self, count: uint, sep: Sep) -> CharSplitsN<'a, Sep>;
    fn split_terminator<'a, Sep: CharEq>(&'a self, sep: Sep) -> CharSplits<'a, Sep>;
    fn rsplitn<'a, Sep: CharEq>(&'a self, count: uint, sep: Sep) -> CharSplitsN<'a, Sep>;
    fn match_indices<'a>(&'a self, sep: &'a str) -> MatchIndices<'a>;
    fn split_str<'a>(&'a self, &'a str) -> StrSplits<'a>;
    fn lines<'a>(&'a self) -> Lines<'a>;
    fn lines_any<'a>(&'a self) -> LinesAny<'a>;
    fn char_len(&self) -> uint;
    fn slice<'a>(&'a self, begin: uint, end: uint) -> &'a str;
    fn slice_from<'a>(&'a self, begin: uint) -> &'a str;
    fn slice_to<'a>(&'a self, end: uint) -> &'a str;
    fn slice_chars<'a>(&'a self, begin: uint, end: uint) -> &'a str;
    unsafe fn slice_unchecked<'a>(&'a self, begin: uint, end: uint) -> &'a str;
    fn starts_with(&self, needle: &str) -> bool;
    fn ends_with(&self, needle: &str) -> bool;
    fn trim_chars<'a, C: CharEq>(&'a self, to_trim: C) -> &'a str;
    fn trim_left_chars<'a, C: CharEq>(&'a self, to_trim: C) -> &'a str;
    fn trim_right_chars<'a, C: CharEq>(&'a self, to_trim: C) -> &'a str;
    fn is_char_boundary(&self, index: uint) -> bool;
    fn char_range_at(&self, start: uint) -> CharRange;
    fn char_range_at_reverse(&self, start: uint) -> CharRange;
    fn char_at(&self, i: uint) -> char;
    fn char_at_reverse(&self, i: uint) -> char;
    fn as_bytes<'a>(&'a self) -> &'a [u8];
    fn find<C: CharEq>(&self, search: C) -> Option<uint>;
    fn rfind<C: CharEq>(&self, search: C) -> Option<uint>;
    fn find_str(&self, &str) -> Option<uint>;
    fn slice_shift_char<'a>(&'a self) -> Option<(char, &'a str)>;
    fn subslice_offset(&self, inner: &str) -> uint;
    fn as_ptr(&self) -> *const u8;
    fn len(&self) -> uint;
    fn is_empty(&self) -> bool;
}

#[inline(never)]
fn slice_error_fail(s: &str, begin: uint, end: uint) -> ! {
    assert!(begin <= end);
    panic!("index {} and/or {} in `{}` do not lie on character boundary",
          begin, end, s);
}

impl StrExt for str {
    #[inline]
    fn contains(&self, needle: &str) -> bool {
        self.find_str(needle).is_some()
    }

    #[inline]
    fn contains_char(&self, needle: char) -> bool {
        self.find(needle).is_some()
    }

    #[inline]
    fn chars(&self) -> Chars {
        Chars{iter: self.as_bytes().iter()}
    }

    #[inline]
    fn bytes(&self) -> Bytes {
        fn deref(&x: &u8) -> u8 { x }

        Bytes { inner: self.as_bytes().iter().map(BytesFn(deref)) }
    }

    #[inline]
    fn char_indices(&self) -> CharIndices {
        CharIndices { front_offset: 0, iter: self.chars() }
    }

    #[inline]
    fn split<Sep: CharEq>(&self, sep: Sep) -> CharSplits<Sep> {
        CharSplits {
            string: self,
            only_ascii: sep.only_ascii(),
            sep: sep,
            allow_trailing_empty: true,
            finished: false,
        }
    }

    #[inline]
    fn splitn<Sep: CharEq>(&self, count: uint, sep: Sep)
        -> CharSplitsN<Sep> {
        CharSplitsN {
            iter: self.split(sep),
            count: count,
            invert: false,
        }
    }

    #[inline]
    fn split_terminator<Sep: CharEq>(&self, sep: Sep)
        -> CharSplits<Sep> {
        CharSplits {
            allow_trailing_empty: false,
            ..self.split(sep)
        }
    }

    #[inline]
    fn rsplitn<Sep: CharEq>(&self, count: uint, sep: Sep)
        -> CharSplitsN<Sep> {
        CharSplitsN {
            iter: self.split(sep),
            count: count,
            invert: true,
        }
    }

    #[inline]
    fn match_indices<'a>(&'a self, sep: &'a str) -> MatchIndices<'a> {
        assert!(!sep.is_empty());
        MatchIndices {
            haystack: self,
            needle: sep,
            searcher: Searcher::new(self.as_bytes(), sep.as_bytes())
        }
    }

    #[inline]
    fn split_str<'a>(&'a self, sep: &'a str) -> StrSplits<'a> {
        StrSplits {
            it: self.match_indices(sep),
            last_end: 0,
            finished: false
        }
    }

    #[inline]
    fn lines(&self) -> Lines {
        Lines { inner: self.split_terminator('\n') }
    }

    fn lines_any(&self) -> LinesAny {
        fn f(line: &str) -> &str {
            let l = line.len();
            if l > 0 && line.as_bytes()[l - 1] == b'\r' { line.slice(0, l - 1) }
            else { line }
        }

        let f: fn(&str) -> &str = f; // coerce to fn pointer
        LinesAny { inner: self.lines().map(f) }
    }

    #[inline]
    fn char_len(&self) -> uint { self.chars().count() }

    #[inline]
    fn slice(&self, begin: uint, end: uint) -> &str {
        // is_char_boundary checks that the index is in [0, .len()]
        if begin <= end &&
           self.is_char_boundary(begin) &&
           self.is_char_boundary(end) {
            unsafe { self.slice_unchecked(begin, end) }
        } else {
            slice_error_fail(self, begin, end)
        }
    }

    #[inline]
    fn slice_from(&self, begin: uint) -> &str {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(begin) {
            unsafe { self.slice_unchecked(begin, self.len()) }
        } else {
            slice_error_fail(self, begin, self.len())
        }
    }

    #[inline]
    fn slice_to(&self, end: uint) -> &str {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(end) {
            unsafe { self.slice_unchecked(0, end) }
        } else {
            slice_error_fail(self, 0, end)
        }
    }

    fn slice_chars(&self, begin: uint, end: uint) -> &str {
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
            (None, _) => panic!("slice_chars: `begin` is beyond end of string"),
            (_, None) => panic!("slice_chars: `end` is beyond end of string"),
            (Some(a), Some(b)) => unsafe { self.slice_unchecked(a, b) }
        }
    }

    #[inline]
    unsafe fn slice_unchecked(&self, begin: uint, end: uint) -> &str {
        mem::transmute(Slice {
            data: self.as_ptr().offset(begin as int),
            len: end - begin,
        })
    }

    #[inline]
    fn starts_with(&self, needle: &str) -> bool {
        let n = needle.len();
        self.len() >= n && needle.as_bytes() == self.as_bytes()[..n]
    }

    #[inline]
    fn ends_with(&self, needle: &str) -> bool {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle.as_bytes() == self.as_bytes()[m-n..]
    }

    #[inline]
    fn trim_chars<C: CharEq>(&self, mut to_trim: C) -> &str {
        let cur = match self.find(|&mut: c: char| !to_trim.matches(c)) {
            None => "",
            Some(i) => unsafe { self.slice_unchecked(i, self.len()) }
        };
        match cur.rfind(|&mut: c: char| !to_trim.matches(c)) {
            None => "",
            Some(i) => {
                let right = cur.char_range_at(i).next;
                unsafe { cur.slice_unchecked(0, right) }
            }
        }
    }

    #[inline]
    fn trim_left_chars<C: CharEq>(&self, mut to_trim: C) -> &str {
        match self.find(|&mut: c: char| !to_trim.matches(c)) {
            None => "",
            Some(first) => unsafe { self.slice_unchecked(first, self.len()) }
        }
    }

    #[inline]
    fn trim_right_chars<C: CharEq>(&self, mut to_trim: C) -> &str {
        match self.rfind(|&mut: c: char| !to_trim.matches(c)) {
            None => "",
            Some(last) => {
                let next = self.char_range_at(last).next;
                unsafe { self.slice_unchecked(0u, next) }
            }
        }
    }

    #[inline]
    fn is_char_boundary(&self, index: uint) -> bool {
        if index == self.len() { return true; }
        match self.as_bytes().get(index) {
            None => false,
            Some(&b) => b < 128u8 || b >= 192u8,
        }
    }

    #[inline]
    fn char_range_at(&self, i: uint) -> CharRange {
        if self.as_bytes()[i] < 128u8 {
            return CharRange {ch: self.as_bytes()[i] as char, next: i + 1 };
        }

        // Multibyte case is a fn to allow char_range_at to inline cleanly
        fn multibyte_char_range_at(s: &str, i: uint) -> CharRange {
            let mut val = s.as_bytes()[i] as u32;
            let w = UTF8_CHAR_WIDTH[val as uint] as uint;
            assert!((w != 0));

            val = utf8_first_byte!(val, w);
            val = utf8_acc_cont_byte!(val, s.as_bytes()[i + 1]);
            if w > 2 { val = utf8_acc_cont_byte!(val, s.as_bytes()[i + 2]); }
            if w > 3 { val = utf8_acc_cont_byte!(val, s.as_bytes()[i + 3]); }

            return CharRange {ch: unsafe { mem::transmute(val) }, next: i + w};
        }

        return multibyte_char_range_at(self, i);
    }

    #[inline]
    fn char_range_at_reverse(&self, start: uint) -> CharRange {
        let mut prev = start;

        prev = prev.saturating_sub(1);
        if self.as_bytes()[prev] < 128 {
            return CharRange{ch: self.as_bytes()[prev] as char, next: prev}
        }

        // Multibyte case is a fn to allow char_range_at_reverse to inline cleanly
        fn multibyte_char_range_at_reverse(s: &str, mut i: uint) -> CharRange {
            // while there is a previous byte == 10......
            while i > 0 && s.as_bytes()[i] & !CONT_MASK == TAG_CONT_U8 {
                i -= 1u;
            }

            let mut val = s.as_bytes()[i] as u32;
            let w = UTF8_CHAR_WIDTH[val as uint] as uint;
            assert!((w != 0));

            val = utf8_first_byte!(val, w);
            val = utf8_acc_cont_byte!(val, s.as_bytes()[i + 1]);
            if w > 2 { val = utf8_acc_cont_byte!(val, s.as_bytes()[i + 2]); }
            if w > 3 { val = utf8_acc_cont_byte!(val, s.as_bytes()[i + 3]); }

            return CharRange {ch: unsafe { mem::transmute(val) }, next: i};
        }

        return multibyte_char_range_at_reverse(self, prev);
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
    fn as_bytes(&self) -> &[u8] {
        unsafe { mem::transmute(self) }
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
    fn slice_shift_char(&self) -> Option<(char, &str)> {
        if self.is_empty() {
            None
        } else {
            let CharRange {ch, next} = self.char_range_at(0u);
            let next_s = unsafe { self.slice_unchecked(next, self.len()) };
            Some((ch, next_s))
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
    fn len(&self) -> uint { self.repr().len }

    #[inline]
    fn is_empty(&self) -> bool { self.len() == 0 }
}

#[stable]
impl<'a> Default for &'a str {
    #[stable]
    fn default() -> &'a str { "" }
}

impl<'a> Iterator<&'a str> for Lines<'a> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}
impl<'a> DoubleEndedIterator<&'a str> for Lines<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> { self.inner.next_back() }
}
impl<'a> Iterator<&'a str> for LinesAny<'a> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}
impl<'a> DoubleEndedIterator<&'a str> for LinesAny<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> { self.inner.next_back() }
}
impl<'a> Iterator<u8> for Bytes<'a> {
    #[inline]
    fn next(&mut self) -> Option<u8> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}
impl<'a> DoubleEndedIterator<u8> for Bytes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<u8> { self.inner.next_back() }
}
impl<'a> ExactSizeIterator<u8> for Bytes<'a> {}
