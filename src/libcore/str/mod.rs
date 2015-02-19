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
use cmp::{self, Eq};
use default::Default;
use error::Error;
use fmt;
use iter::ExactSizeIterator;
use iter::{Map, Iterator, IteratorExt, DoubleEndedIterator};
use marker::Sized;
use mem;
use num::Int;
use ops::{Fn, FnMut};
use option::Option::{self, None, Some};
use ptr::PtrExt;
use raw::{Repr, Slice};
use result::Result::{self, Ok, Err};
use slice::{self, SliceExt};
use usize;

macro_rules! delegate_iter {
    (exact $te:ty : $ti:ty) => {
        delegate_iter!{$te : $ti}
        impl<'a> ExactSizeIterator for $ti {
            #[inline]
            fn len(&self) -> usize {
                self.0.len()
            }
        }
    };
    ($te:ty : $ti:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> Iterator for $ti {
            type Item = $te;

            #[inline]
            fn next(&mut self) -> Option<$te> {
                self.0.next()
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a> DoubleEndedIterator for $ti {
            #[inline]
            fn next_back(&mut self) -> Option<$te> {
                self.0.next_back()
            }
        }
    };
    (pattern $te:ty : $ti:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, P: CharEq> Iterator for $ti {
            type Item = $te;

            #[inline]
            fn next(&mut self) -> Option<$te> {
                self.0.next()
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, P: CharEq> DoubleEndedIterator for $ti {
            #[inline]
            fn next_back(&mut self) -> Option<$te> {
                self.0.next_back()
            }
        }
    };
    (pattern forward $te:ty : $ti:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, P: CharEq> Iterator for $ti {
            type Item = $te;

            #[inline]
            fn next(&mut self) -> Option<$te> {
                self.0.next()
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }
    }
}

/// A trait to abstract the idea of creating a new instance of a type from a
/// string.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait FromStr {
    /// The associated error which can be returned from parsing.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Err;

    /// Parses a string `s` to return an optional value of this type. If the
    /// string is ill-formatted, the None is returned.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_str(s: &str) -> Result<Self, Self::Err>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for bool {
    type Err = ParseBoolError;

    /// Parse a `bool` from a string.
    ///
    /// Yields an `Option<bool>`, because `s` may or may not actually be
    /// parseable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!("true".parse(), Ok(true));
    /// assert_eq!("false".parse(), Ok(false));
    /// assert!("not even a boolean".parse::<bool>().is_err());
    /// ```
    #[inline]
    fn from_str(s: &str) -> Result<bool, ParseBoolError> {
        match s {
            "true"  => Ok(true),
            "false" => Ok(false),
            _       => Err(ParseBoolError { _priv: () }),
        }
    }
}

/// An error returned when parsing a `bool` from a string fails.
#[derive(Debug, Clone, PartialEq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ParseBoolError { _priv: () }

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseBoolError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "provided string was not `true` or `false`".fmt(f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for ParseBoolError {
    fn description(&self) -> &str { "failed to parse bool" }
}

/*
Section: Creating a string
*/

/// Errors which can occur when attempting to interpret a byte slice as a `str`.
#[derive(Copy, Eq, PartialEq, Clone, Debug)]
#[unstable(feature = "core",
           reason = "error enumeration recently added and definitions may be refined")]
pub enum Utf8Error {
    /// An invalid byte was detected at the byte offset given.
    ///
    /// The offset is guaranteed to be in bounds of the slice in question, and
    /// the byte at the specified offset was the first invalid byte in the
    /// sequence detected.
    InvalidByte(usize),

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
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_utf8(v: &[u8]) -> Result<&str, Utf8Error> {
    try!(run_utf8_validation_iterator(&mut v.iter()));
    Ok(unsafe { from_utf8_unchecked(v) })
}

/// Converts a slice of bytes to a string slice without checking
/// that the string contains valid UTF-8.
#[stable(feature = "rust1", since = "1.0.0")]
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
#[unstable(feature = "core")]
#[deprecated(since = "1.0.0",
             reason = "use std::ffi::c_str_to_bytes + str::from_utf8")]
pub unsafe fn from_c_str(s: *const i8) -> &'static str {
    let s = s as *const u8;
    let mut len = 0;
    while *s.offset(len as isize) != 0 {
        len += 1;
    }
    let v: &'static [u8] = ::mem::transmute(Slice { data: s, len: len });
    from_utf8(v).ok().expect("from_c_str passed invalid utf-8 data")
}

/// Something that can be used to compare against a character
#[unstable(feature = "core",
           reason = "definition may change as pattern-related methods are stabilized")]
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
    fn only_ascii(&self) -> bool { (*self as u32) < 128 }
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
        self.iter().any(|&m| { let mut m = m; m.matches(c) })
    }

    #[inline]
    fn only_ascii(&self) -> bool {
        self.iter().all(|m| m.only_ascii())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for Utf8Error {
    fn description(&self) -> &str {
        match *self {
            Utf8Error::TooShort => "invalid utf-8: not enough bytes",
            Utf8Error::InvalidByte(..) => "invalid utf-8: corrupt contents",
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Utf8Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Utf8Error::InvalidByte(n) => {
                write!(f, "invalid utf-8: invalid byte at index {}", n)
            }
            Utf8Error::TooShort => {
                write!(f, "invalid utf-8: byte slice too short")
            }
        }
    }
}

/*
Section: Iterators
*/

/// Iterator for the char (representing *Unicode Scalar Values*) of a string
///
/// Created with the method `.chars()`.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
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

/// Reads the next code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
#[unstable(feature = "core")]
pub fn next_code_point(bytes: &mut slice::Iter<u8>) -> Option<u32> {
    // Decode UTF-8
    let x = match bytes.next() {
        None => return None,
        Some(&next_byte) if next_byte < 128 => return Some(next_byte as u32),
        Some(&next_byte) => next_byte,
    };

    // Multibyte case follows
    // Decode from a byte combination out of: [[[x y] z] w]
    // NOTE: Performance is sensitive to the exact formulation here
    let init = utf8_first_byte!(x, 2);
    let y = unwrap_or_0(bytes.next());
    let mut ch = utf8_acc_cont_byte!(init, y);
    if x >= 0xE0 {
        // [[x y z] w] case
        // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
        let z = unwrap_or_0(bytes.next());
        let y_z = utf8_acc_cont_byte!((y & CONT_MASK) as u32, z);
        ch = init << 12 | y_z;
        if x >= 0xF0 {
            // [x y z w] case
            // use only the lower 3 bits of `init`
            let w = unwrap_or_0(bytes.next());
            ch = (init & 7) << 18 | utf8_acc_cont_byte!(y_z, w);
        }
    }

    Some(ch)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Chars<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        next_code_point(&mut self.iter).map(|ch| {
            // str invariant says `ch` is a valid Unicode Scalar Value
            unsafe {
                mem::transmute(ch)
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (len, _) = self.iter.size_hint();
        (len.saturating_add(3) / 4, Some(len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Chars<'a> {
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
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct CharIndices<'a> {
    front_offset: usize,
    iter: Chars<'a>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for CharIndices<'a> {
    type Item = (usize, char);

    #[inline]
    fn next(&mut self) -> Option<(usize, char)> {
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for CharIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, char)> {
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
///
/// Created with `StrExt::bytes`
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Bytes<'a>(Map<slice::Iter<'a, u8>, BytesDeref>);
delegate_iter!{exact u8 : Bytes<'a>}

/// A temporary fn new type that ensures that the `Bytes` iterator
/// is cloneable.
#[derive(Copy, Clone)]
struct BytesDeref;

impl<'a> Fn<(&'a u8,)> for BytesDeref {
    type Output = u8;

    #[inline]
    extern "rust-call" fn call(&self, (ptr,): (&'a u8,)) -> u8 {
        *ptr
    }
}

/// An iterator over the substrings of a string, separated by `sep`.
#[derive(Clone)]
struct CharSplits<'a, Sep> {
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
#[derive(Clone)]
struct CharSplitsN<'a, Sep> {
    iter: CharSplits<'a, Sep>,
    /// The number of splits remaining
    count: usize,
    invert: bool,
}

/// An iterator over the lines of a string, separated by `\n`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Lines<'a> {
    inner: CharSplits<'a, char>,
}

/// An iterator over the lines of a string, separated by either `\n` or (`\r\n`).
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LinesAny<'a> {
    inner: Map<Lines<'a>, fn(&str) -> &str>,
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, Sep: CharEq> Iterator for CharSplits<'a, Sep> {
    type Item = &'a str;

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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, Sep: CharEq> DoubleEndedIterator for CharSplits<'a, Sep> {
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, Sep: CharEq> Iterator for CharSplitsN<'a, Sep> {
    type Item = &'a str;

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
#[derive(Clone)]
struct NaiveSearcher {
    position: usize
}

impl NaiveSearcher {
    fn new() -> NaiveSearcher {
        NaiveSearcher { position: 0 }
    }

    fn next(&mut self, haystack: &[u8], needle: &[u8]) -> Option<(usize, usize)> {
        while self.position + needle.len() <= haystack.len() {
            if &haystack[self.position .. self.position + needle.len()] == needle {
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
#[derive(Clone)]
struct TwoWaySearcher {
    // constants
    crit_pos: usize,
    period: usize,
    byteset: u64,

    // variables
    position: usize,
    memory: usize
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
        let (crit_pos_false, period_false) = TwoWaySearcher::maximal_suffix(needle, false);
        let (crit_pos_true, period_true) = TwoWaySearcher::maximal_suffix(needle, true);

        let (crit_pos, period) =
            if crit_pos_false > crit_pos_true {
                (crit_pos_false, period_false)
            } else {
                (crit_pos_true, period_true)
            };

        // This isn't in the original algorithm, as far as I'm aware.
        let byteset = needle.iter()
                            .fold(0, |a, &b| (1 << ((b & 0x3f) as usize)) | a);

        // A particularly readable explanation of what's going on here can be found
        // in Crochemore and Rytter's book "Text Algorithms", ch 13. Specifically
        // see the code for "Algorithm CP" on p. 323.
        //
        // What's going on is we have some critical factorization (u, v) of the
        // needle, and we want to determine whether u is a suffix of
        // &v[..period]. If it is, we use "Algorithm CP1". Otherwise we use
        // "Algorithm CP2", which is optimized for when the period of the needle
        // is large.
        if &needle[..crit_pos] == &needle[period.. period + crit_pos] {
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
                memory: usize::MAX // Dummy value to signify that the period is long
            }
        }
    }

    // One of the main ideas of Two-Way is that we factorize the needle into
    // two halves, (u, v), and begin trying to find v in the haystack by scanning
    // left to right. If v matches, we try to match u by scanning right to left.
    // How far we can jump when we encounter a mismatch is all based on the fact
    // that (u, v) is a critical factorization for the needle.
    #[inline]
    fn next(&mut self, haystack: &[u8], needle: &[u8], long_period: bool)
            -> Option<(usize, usize)> {
        'search: loop {
            // Check that we have room to search in
            if self.position + needle.len() > haystack.len() {
                return None;
            }

            // Quickly skip by large portions unrelated to our substring
            if (self.byteset >>
                    ((haystack[self.position + needle.len() - 1] & 0x3f)
                     as usize)) & 1 == 0 {
                self.position += needle.len();
                if !long_period {
                    self.memory = 0;
                }
                continue 'search;
            }

            // See if the right part of the needle matches
            let start = if long_period { self.crit_pos }
                        else { cmp::max(self.crit_pos, self.memory) };
            for i in start..needle.len() {
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
            for i in (start..self.crit_pos).rev() {
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
    fn maximal_suffix(arr: &[u8], reversed: bool) -> (usize, usize) {
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
#[derive(Clone)]
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
            if searcher.memory == usize::MAX { // If the period is long
                TwoWayLong(searcher)
            } else {
                TwoWay(searcher)
            }
        }
    }
}

/// An iterator over the start and end indices of the matches of a
/// substring within a larger string
#[derive(Clone)]
#[unstable(feature = "core", reason = "type may be removed")]
pub struct MatchIndices<'a> {
    // constants
    haystack: &'a str,
    needle: &'a str,
    searcher: Searcher
}

/// An iterator over the substrings of a string separated by a given
/// search string
#[derive(Clone)]
#[unstable(feature = "core", reason = "type may be removed")]
pub struct SplitStr<'a> {
    it: MatchIndices<'a>,
    last_end: usize,
    finished: bool
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for MatchIndices<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for SplitStr<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.finished { return None; }

        match self.it.next() {
            Some((from, to)) => {
                let ret = Some(&self.it.haystack[self.last_end .. from]);
                self.last_end = to;
                ret
            }
            None => {
                self.finished = true;
                Some(&self.it.haystack[self.last_end .. self.it.haystack.len()])
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
    // NOTE: In theory n should be libc::size_t and not usize, but libc is not available here
    #[allow(improper_ctypes)]
    extern { fn memcmp(s1: *const i8, s2: *const i8, n: usize) -> i32; }
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
        let old = iter.clone();

        // restore the iterator we had at the start of this codepoint.
        macro_rules! err { () => {{
            *iter = old.clone();
            return Err(Utf8Error::InvalidByte(whole.len() - iter.as_slice().len()))
        }}}

        macro_rules! next { () => {
            match iter.next() {
                Some(a) => *a,
                // we needed data, but there was none: error!
                None => return Err(Utf8Error::TooShort),
            }
        }}

        let first = match iter.next() {
            Some(&b) => b,
            // we're at the end of the iterator and a codepoint
            // boundary at the same time, so this string is valid.
            None => return Ok(())
        };

        // ASCII characters are always valid, so only large
        // bytes need more examination.
        if first >= 128 {
            let w = UTF8_CHAR_WIDTH[first as usize] as usize;
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

/// Struct that contains a `char` and the index of the first byte of
/// the next `char` in a string.  This can be used as a data structure
/// for iterating over the UTF-8 bytes of a string.
#[derive(Copy)]
#[unstable(feature = "core",
           reason = "naming is uncertain with container conventions")]
pub struct CharRange {
    /// Current `char`
    pub ch: char,
    /// Index of the first byte of the next `char`
    pub next: usize,
}

/// Mask of the value bits of a continuation byte
const CONT_MASK: u8 = 0b0011_1111u8;
/// Value of the tag bits (tag mask is !CONT_MASK) of a continuation byte
const TAG_CONT_U8: u8 = 0b1000_0000u8;

/*
Section: Trait implementations
*/

mod traits {
    use cmp::{Ordering, Ord, PartialEq, PartialOrd, Eq};
    use cmp::Ordering::{Less, Equal, Greater};
    use iter::IteratorExt;
    use option::Option;
    use option::Option::Some;
    use ops;
    use str::{StrExt, eq_slice};

    #[stable(feature = "rust1", since = "1.0.0")]
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

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialEq for str {
        #[inline]
        fn eq(&self, other: &str) -> bool {
            eq_slice(self, other)
        }
        #[inline]
        fn ne(&self, other: &str) -> bool { !(*self).eq(other) }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Eq for str {}

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialOrd for str {
        #[inline]
        fn partial_cmp(&self, other: &str) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    /// Returns a slice of the given string from the byte range
    /// [`begin`..`end`).
    ///
    /// This operation is `O(1)`.
    ///
    /// Panics when `begin` and `end` do not point to valid characters
    /// or point beyond the last character of the string.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// assert_eq!(&s[0 .. 1], "L");
    ///
    /// assert_eq!(&s[1 .. 9], "öwe 老");
    ///
    /// // these will panic:
    /// // byte 2 lies within `ö`:
    /// // &s[2 ..3];
    ///
    /// // byte 8 lies within `老`
    /// // &s[1 .. 8];
    ///
    /// // byte 100 is outside the string
    /// // &s[3 .. 100];
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::Range<usize>> for str {
        type Output = str;
        #[inline]
        fn index(&self, index: &ops::Range<usize>) -> &str {
            // is_char_boundary checks that the index is in [0, .len()]
            if index.start <= index.end &&
               self.is_char_boundary(index.start) &&
               self.is_char_boundary(index.end) {
                unsafe { self.slice_unchecked(index.start, index.end) }
            } else {
                super::slice_error_fail(self, index.start, index.end)
            }
        }
    }

    /// Returns a slice of the string from the beginning to byte
    /// `end`.
    ///
    /// Equivalent to `self[0 .. end]`.
    ///
    /// Panics when `end` does not point to a valid character, or is
    /// out of bounds.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::RangeTo<usize>> for str {
        type Output = str;
        #[inline]
        fn index(&self, index: &ops::RangeTo<usize>) -> &str {
            // is_char_boundary checks that the index is in [0, .len()]
            if self.is_char_boundary(index.end) {
                unsafe { self.slice_unchecked(0, index.end) }
            } else {
                super::slice_error_fail(self, 0, index.end)
            }
        }
    }

    /// Returns a slice of the string from `begin` to its end.
    ///
    /// Equivalent to `self[begin .. self.len()]`.
    ///
    /// Panics when `begin` does not point to a valid character, or is
    /// out of bounds.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::RangeFrom<usize>> for str {
        type Output = str;
        #[inline]
        fn index(&self, index: &ops::RangeFrom<usize>) -> &str {
            // is_char_boundary checks that the index is in [0, .len()]
            if self.is_char_boundary(index.start) {
                unsafe { self.slice_unchecked(index.start, self.len()) }
            } else {
                super::slice_error_fail(self, index.start, self.len())
            }
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::RangeFull> for str {
        type Output = str;
        #[inline]
        fn index(&self, _index: &ops::RangeFull) -> &str {
            self
        }
    }
}

/// Any string that can be represented as a slice
#[unstable(feature = "core",
           reason = "Instead of taking this bound generically, this trait will be \
                     replaced with one of slicing syntax (&foo[..]), deref coercions, or \
                     a more generic conversion trait")]
pub trait Str {
    /// Work with `self` as a slice.
    fn as_slice<'a>(&'a self) -> &'a str;
}

impl Str for str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str { self }
}

impl<'a, S: ?Sized> Str for &'a S where S: Str {
    #[inline]
    fn as_slice(&self) -> &str { Str::as_slice(*self) }
}

/// Return type of `StrExt::split`
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Split<'a, P>(CharSplits<'a, P>);
delegate_iter!{pattern &'a str : Split<'a, P>}

/// Return type of `StrExt::split_terminator`
#[derive(Clone)]
#[unstable(feature = "core",
           reason = "might get removed in favour of a constructor method on Split")]
pub struct SplitTerminator<'a, P>(CharSplits<'a, P>);
delegate_iter!{pattern &'a str : SplitTerminator<'a, P>}

/// Return type of `StrExt::splitn`
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitN<'a, P>(CharSplitsN<'a, P>);
delegate_iter!{pattern forward &'a str : SplitN<'a, P>}

/// Return type of `StrExt::rsplitn`
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitN<'a, P>(CharSplitsN<'a, P>);
delegate_iter!{pattern forward &'a str : RSplitN<'a, P>}

/// Methods for string slices
#[allow(missing_docs)]
pub trait StrExt {
    // NB there are no docs here are they're all located on the StrExt trait in
    // libcollections, not here.

    fn contains(&self, pat: &str) -> bool;
    fn contains_char<P: CharEq>(&self, pat: P) -> bool;
    fn chars<'a>(&'a self) -> Chars<'a>;
    fn bytes<'a>(&'a self) -> Bytes<'a>;
    fn char_indices<'a>(&'a self) -> CharIndices<'a>;
    fn split<'a, P: CharEq>(&'a self, pat: P) -> Split<'a, P>;
    fn splitn<'a, P: CharEq>(&'a self, count: usize, pat: P) -> SplitN<'a, P>;
    fn split_terminator<'a, P: CharEq>(&'a self, pat: P) -> SplitTerminator<'a, P>;
    fn rsplitn<'a, P: CharEq>(&'a self, count: usize, pat: P) -> RSplitN<'a, P>;
    fn match_indices<'a>(&'a self, sep: &'a str) -> MatchIndices<'a>;
    fn split_str<'a>(&'a self, pat: &'a str) -> SplitStr<'a>;
    fn lines<'a>(&'a self) -> Lines<'a>;
    fn lines_any<'a>(&'a self) -> LinesAny<'a>;
    fn char_len(&self) -> usize;
    fn slice_chars<'a>(&'a self, begin: usize, end: usize) -> &'a str;
    unsafe fn slice_unchecked<'a>(&'a self, begin: usize, end: usize) -> &'a str;
    fn starts_with(&self, pat: &str) -> bool;
    fn ends_with(&self, pat: &str) -> bool;
    fn trim_matches<'a, P: CharEq>(&'a self, pat: P) -> &'a str;
    fn trim_left_matches<'a, P: CharEq>(&'a self, pat: P) -> &'a str;
    fn trim_right_matches<'a, P: CharEq>(&'a self, pat: P) -> &'a str;
    fn is_char_boundary(&self, index: usize) -> bool;
    fn char_range_at(&self, start: usize) -> CharRange;
    fn char_range_at_reverse(&self, start: usize) -> CharRange;
    fn char_at(&self, i: usize) -> char;
    fn char_at_reverse(&self, i: usize) -> char;
    fn as_bytes<'a>(&'a self) -> &'a [u8];
    fn find<P: CharEq>(&self, pat: P) -> Option<usize>;
    fn rfind<P: CharEq>(&self, pat: P) -> Option<usize>;
    fn find_str(&self, pat: &str) -> Option<usize>;
    fn slice_shift_char<'a>(&'a self) -> Option<(char, &'a str)>;
    fn subslice_offset(&self, inner: &str) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn parse<T: FromStr>(&self) -> Result<T, T::Err>;
}

#[inline(never)]
fn slice_error_fail(s: &str, begin: usize, end: usize) -> ! {
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
    fn contains_char<P: CharEq>(&self, pat: P) -> bool {
        self.find(pat).is_some()
    }

    #[inline]
    fn chars(&self) -> Chars {
        Chars{iter: self.as_bytes().iter()}
    }

    #[inline]
    fn bytes(&self) -> Bytes {
        Bytes(self.as_bytes().iter().map(BytesDeref))
    }

    #[inline]
    fn char_indices(&self) -> CharIndices {
        CharIndices { front_offset: 0, iter: self.chars() }
    }

    #[inline]
    fn split<P: CharEq>(&self, pat: P) -> Split<P> {
        Split(CharSplits {
            string: self,
            only_ascii: pat.only_ascii(),
            sep: pat,
            allow_trailing_empty: true,
            finished: false,
        })
    }

    #[inline]
    fn splitn<P: CharEq>(&self, count: usize, pat: P) -> SplitN<P> {
        SplitN(CharSplitsN {
            iter: self.split(pat).0,
            count: count,
            invert: false,
        })
    }

    #[inline]
    fn split_terminator<P: CharEq>(&self, pat: P) -> SplitTerminator<P> {
        SplitTerminator(CharSplits {
            allow_trailing_empty: false,
            ..self.split(pat).0
        })
    }

    #[inline]
    fn rsplitn<P: CharEq>(&self, count: usize, pat: P) -> RSplitN<P> {
        RSplitN(CharSplitsN {
            iter: self.split(pat).0,
            count: count,
            invert: true,
        })
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
    fn split_str<'a>(&'a self, sep: &'a str) -> SplitStr<'a> {
        SplitStr {
            it: self.match_indices(sep),
            last_end: 0,
            finished: false
        }
    }

    #[inline]
    fn lines(&self) -> Lines {
        Lines { inner: self.split_terminator('\n').0 }
    }

    fn lines_any(&self) -> LinesAny {
        fn f(line: &str) -> &str {
            let l = line.len();
            if l > 0 && line.as_bytes()[l - 1] == b'\r' { &line[0 .. l - 1] }
            else { line }
        }

        let f: fn(&str) -> &str = f; // coerce to fn pointer
        LinesAny { inner: self.lines().map(f) }
    }

    #[inline]
    fn char_len(&self) -> usize { self.chars().count() }

    fn slice_chars(&self, begin: usize, end: usize) -> &str {
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
    unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        mem::transmute(Slice {
            data: self.as_ptr().offset(begin as isize),
            len: end - begin,
        })
    }

    #[inline]
    fn starts_with(&self, needle: &str) -> bool {
        let n = needle.len();
        self.len() >= n && needle.as_bytes() == &self.as_bytes()[..n]
    }

    #[inline]
    fn ends_with(&self, needle: &str) -> bool {
        let (m, n) = (self.len(), needle.len());
        m >= n && needle.as_bytes() == &self.as_bytes()[m-n..]
    }

    #[inline]
    fn trim_matches<P: CharEq>(&self, mut pat: P) -> &str {
        let cur = match self.find(|c: char| !pat.matches(c)) {
            None => "",
            Some(i) => unsafe { self.slice_unchecked(i, self.len()) }
        };
        match cur.rfind(|c: char| !pat.matches(c)) {
            None => "",
            Some(i) => {
                let right = cur.char_range_at(i).next;
                unsafe { cur.slice_unchecked(0, right) }
            }
        }
    }

    #[inline]
    fn trim_left_matches<P: CharEq>(&self, mut pat: P) -> &str {
        match self.find(|c: char| !pat.matches(c)) {
            None => "",
            Some(first) => unsafe { self.slice_unchecked(first, self.len()) }
        }
    }

    #[inline]
    fn trim_right_matches<P: CharEq>(&self, mut pat: P) -> &str {
        match self.rfind(|c: char| !pat.matches(c)) {
            None => "",
            Some(last) => {
                let next = self.char_range_at(last).next;
                unsafe { self.slice_unchecked(0, next) }
            }
        }
    }

    #[inline]
    fn is_char_boundary(&self, index: usize) -> bool {
        if index == self.len() { return true; }
        match self.as_bytes().get(index) {
            None => false,
            Some(&b) => b < 128u8 || b >= 192u8,
        }
    }

    #[inline]
    fn char_range_at(&self, i: usize) -> CharRange {
        let (c, n) = char_range_at_raw(self.as_bytes(), i);
        CharRange { ch: unsafe { mem::transmute(c) }, next: n }
    }

    #[inline]
    fn char_range_at_reverse(&self, start: usize) -> CharRange {
        let mut prev = start;

        prev = prev.saturating_sub(1);
        if self.as_bytes()[prev] < 128 {
            return CharRange{ch: self.as_bytes()[prev] as char, next: prev}
        }

        // Multibyte case is a fn to allow char_range_at_reverse to inline cleanly
        fn multibyte_char_range_at_reverse(s: &str, mut i: usize) -> CharRange {
            // while there is a previous byte == 10......
            while i > 0 && s.as_bytes()[i] & !CONT_MASK == TAG_CONT_U8 {
                i -= 1;
            }

            let mut val = s.as_bytes()[i] as u32;
            let w = UTF8_CHAR_WIDTH[val as usize] as usize;
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
    fn char_at(&self, i: usize) -> char {
        self.char_range_at(i).ch
    }

    #[inline]
    fn char_at_reverse(&self, i: usize) -> char {
        self.char_range_at_reverse(i).ch
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe { mem::transmute(self) }
    }

    fn find<P: CharEq>(&self, mut pat: P) -> Option<usize> {
        if pat.only_ascii() {
            self.bytes().position(|b| pat.matches(b as char))
        } else {
            for (index, c) in self.char_indices() {
                if pat.matches(c) { return Some(index); }
            }
            None
        }
    }

    fn rfind<P: CharEq>(&self, mut pat: P) -> Option<usize> {
        if pat.only_ascii() {
            self.bytes().rposition(|b| pat.matches(b as char))
        } else {
            for (index, c) in self.char_indices().rev() {
                if pat.matches(c) { return Some(index); }
            }
            None
        }
    }

    fn find_str(&self, needle: &str) -> Option<usize> {
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
            let CharRange {ch, next} = self.char_range_at(0);
            let next_s = unsafe { self.slice_unchecked(next, self.len()) };
            Some((ch, next_s))
        }
    }

    fn subslice_offset(&self, inner: &str) -> usize {
        let a_start = self.as_ptr() as usize;
        let a_end = a_start + self.len();
        let b_start = inner.as_ptr() as usize;
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
    fn len(&self) -> usize { self.repr().len }

    #[inline]
    fn is_empty(&self) -> bool { self.len() == 0 }

    #[inline]
    fn parse<T: FromStr>(&self) -> Result<T, T::Err> { FromStr::from_str(self) }
}

/// Pluck a code point out of a UTF-8-like byte slice and return the
/// index of the next code point.
#[inline]
#[unstable(feature = "core")]
pub fn char_range_at_raw(bytes: &[u8], i: usize) -> (u32, usize) {
    if bytes[i] < 128u8 {
        return (bytes[i] as u32, i + 1);
    }

    // Multibyte case is a fn to allow char_range_at to inline cleanly
    fn multibyte_char_range_at(bytes: &[u8], i: usize) -> (u32, usize) {
        let mut val = bytes[i] as u32;
        let w = UTF8_CHAR_WIDTH[val as usize] as usize;
        assert!((w != 0));

        val = utf8_first_byte!(val, w);
        val = utf8_acc_cont_byte!(val, bytes[i + 1]);
        if w > 2 { val = utf8_acc_cont_byte!(val, bytes[i + 2]); }
        if w > 3 { val = utf8_acc_cont_byte!(val, bytes[i + 3]); }

        return (val, i + w);
    }

    multibyte_char_range_at(bytes, i)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Default for &'a str {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> &'a str { "" }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Lines<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Lines<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> { self.inner.next_back() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for LinesAny<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for LinesAny<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> { self.inner.next_back() }
}
