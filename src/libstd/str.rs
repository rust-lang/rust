// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * String manipulation
 *
 * Strings are a packed UTF-8 representation of text, stored as null
 * terminated buffers of u8 bytes.  Strings should be indexed in bytes,
 * for efficiency, but UTF-8 unsafe operations should be avoided.
 */

use at_vec;
use cast;
use char;
use char::Char;
use clone::{Clone, DeepClone};
use container::{Container, Mutable};
use iter::Times;
use iterator::{Iterator, FromIterator, Extendable};
use iterator::{Filter, AdditiveIterator, Map};
use iterator::{Invert, DoubleEndedIterator};
use libc;
use num::Zero;
use option::{None, Option, Some};
use ptr;
use ptr::RawPtr;
use to_str::ToStr;
use uint;
use unstable::raw::{Repr, Slice};
use vec;
use vec::{OwnedVector, OwnedCopyableVector, ImmutableVector, MutableVector};

/*
Section: Conditions
*/

condition! {
    not_utf8: (~str) -> ~str;
}

/*
Section: Creating a string
*/

/// Convert a vector of bytes to a new UTF-8 string
///
/// # Failure
///
/// Raises the `not_utf8` condition if invalid UTF-8
pub fn from_bytes(vv: &[u8]) -> ~str {
    use str::not_utf8::cond;

    if !is_utf8(vv) {
        let first_bad_byte = *vv.iter().find(|&b| !is_utf8([*b])).unwrap();
        cond.raise(fmt!("from_bytes: input is not UTF-8; first bad byte is %u",
                        first_bad_byte as uint))
    } else {
        return unsafe { raw::from_bytes(vv) }
    }
}

/// Consumes a vector of bytes to create a new utf-8 string
///
/// # Failure
///
/// Raises the `not_utf8` condition if invalid UTF-8
pub fn from_bytes_owned(vv: ~[u8]) -> ~str {
    use str::not_utf8::cond;

    if !is_utf8(vv) {
        let first_bad_byte = *vv.iter().find(|&b| !is_utf8([*b])).unwrap();
        cond.raise(fmt!("from_bytes: input is not UTF-8; first bad byte is %u",
                        first_bad_byte as uint))
    } else {
        return unsafe { raw::from_bytes_owned(vv) }
    }
}

/// Converts a vector to a string slice without performing any allocations.
///
/// Once the slice has been validated as utf-8, it is transmuted in-place and
/// returned as a '&str' instead of a '&[u8]'
///
/// # Failure
///
/// Fails if invalid UTF-8
pub fn from_bytes_slice<'a>(v: &'a [u8]) -> &'a str {
    assert!(is_utf8(v));
    unsafe { cast::transmute(v) }
}

impl ToStr for ~str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
}
impl<'self> ToStr for &'self str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
}
impl ToStr for @str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
}

/// Convert a byte to a UTF-8 string
///
/// # Failure
///
/// Fails if invalid UTF-8
pub fn from_byte(b: u8) -> ~str {
    assert!(b < 128u8);
    unsafe { ::cast::transmute(~[b]) }
}

/// Convert a char to a string
pub fn from_char(ch: char) -> ~str {
    let mut buf = ~"";
    buf.push_char(ch);
    buf
}

/// Convert a vector of chars to a string
pub fn from_chars(chs: &[char]) -> ~str {
    let mut buf = ~"";
    buf.reserve(chs.len());
    for ch in chs.iter() {
        buf.push_char(*ch)
    }
    buf
}

#[doc(hidden)]
pub fn push_str(lhs: &mut ~str, rhs: &str) {
    lhs.push_str(rhs)
}

#[allow(missing_doc)]
pub trait StrVector {
    fn concat(&self) -> ~str;
    fn connect(&self, sep: &str) -> ~str;
}

impl<'self, S: Str> StrVector for &'self [S] {
    /// Concatenate a vector of strings.
    fn concat(&self) -> ~str {
        if self.is_empty() { return ~""; }

        let len = self.iter().map(|s| s.as_slice().len()).sum();

        let mut s = with_capacity(len);

        unsafe {
            do s.as_mut_buf |buf, _| {
                let mut buf = buf;
                for ss in self.iter() {
                    do ss.as_slice().as_imm_buf |ssbuf, sslen| {
                        ptr::copy_memory(buf, ssbuf, sslen);
                        buf = buf.offset(sslen as int);
                    }
                }
            }
            raw::set_len(&mut s, len);
        }
        s
    }

    /// Concatenate a vector of strings, placing a given separator between each.
    fn connect(&self, sep: &str) -> ~str {
        if self.is_empty() { return ~""; }

        // concat is faster
        if sep.is_empty() { return self.concat(); }

        // this is wrong without the guarantee that `self` is non-empty
        let len = sep.len() * (self.len() - 1)
            + self.iter().map(|s| s.as_slice().len()).sum();
        let mut s = ~"";
        let mut first = true;

        s.reserve(len);

        unsafe {
            do s.as_mut_buf |buf, _| {
                do sep.as_imm_buf |sepbuf, seplen| {
                    let mut buf = buf;
                    for ss in self.iter() {
                        do ss.as_slice().as_imm_buf |ssbuf, sslen| {
                            if first {
                                first = false;
                            } else {
                                ptr::copy_memory(buf, sepbuf, seplen);
                                buf = buf.offset(seplen as int);
                            }
                            ptr::copy_memory(buf, ssbuf, sslen);
                            buf = buf.offset(sslen as int);
                        }
                    }
                }
            }
            raw::set_len(&mut s, len);
        }
        s
    }
}

/// Something that can be used to compare against a character
pub trait CharEq {
    /// Determine if the splitter should split at the given character
    fn matches(&self, char) -> bool;
    /// Indicate if this is only concerned about ASCII characters,
    /// which can allow for a faster implementation.
    fn only_ascii(&self) -> bool;
}

impl CharEq for char {
    #[inline]
    fn matches(&self, c: char) -> bool { *self == c }

    fn only_ascii(&self) -> bool { (*self as uint) < 128 }
}

impl<'self> CharEq for &'self fn(char) -> bool {
    #[inline]
    fn matches(&self, c: char) -> bool { (*self)(c) }

    fn only_ascii(&self) -> bool { false }
}

impl CharEq for extern "Rust" fn(char) -> bool {
    #[inline]
    fn matches(&self, c: char) -> bool { (*self)(c) }

    fn only_ascii(&self) -> bool { false }
}

impl<'self, C: CharEq> CharEq for &'self [C] {
    #[inline]
    fn matches(&self, c: char) -> bool {
        self.iter().any(|m| m.matches(c))
    }

    fn only_ascii(&self) -> bool {
        self.iter().all(|m| m.only_ascii())
    }
}

/*
Section: Iterators
*/

/// External iterator for a string's characters and their byte offsets.
/// Use with the `std::iterator` module.
#[deriving(Clone)]
pub struct CharOffsetIterator<'self> {
    priv index_front: uint,
    priv index_back: uint,
    priv string: &'self str,
}

impl<'self> Iterator<(uint, char)> for CharOffsetIterator<'self> {
    #[inline]
    fn next(&mut self) -> Option<(uint, char)> {
        if self.index_front < self.index_back {
            let CharRange {ch, next} = self.string.char_range_at(self.index_front);
            let index = self.index_front;
            self.index_front = next;
            Some((index, ch))
        } else {
            None
        }
    }
}

impl<'self> DoubleEndedIterator<(uint, char)> for CharOffsetIterator<'self> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, char)> {
        if self.index_front < self.index_back {
            let CharRange {ch, next} = self.string.char_range_at_reverse(self.index_back);
            self.index_back = next;
            Some((next, ch))
        } else {
            None
        }
    }
}

/// External iterator for a string's characters and their byte offsets in reverse order.
/// Use with the `std::iterator` module.
pub type CharOffsetRevIterator<'self> =
    Invert<CharOffsetIterator<'self>>;

/// External iterator for a string's characters.
/// Use with the `std::iterator` module.
pub type CharIterator<'self> =
    Map<'self, (uint, char), char, CharOffsetIterator<'self>>;

/// External iterator for a string's characters in reverse order.
/// Use with the `std::iterator` module.
pub type CharRevIterator<'self> =
    Invert<Map<'self, (uint, char), char, CharOffsetIterator<'self>>>;

/// External iterator for a string's bytes.
/// Use with the `std::iterator` module.
pub type ByteIterator<'self> =
    Map<'self, &'self u8, u8, vec::VecIterator<'self, u8>>;

/// External iterator for a string's bytes in reverse order.
/// Use with the `std::iterator` module.
pub type ByteRevIterator<'self> =
    Invert<Map<'self, &'self u8, u8, vec::VecIterator<'self, u8>>>;

/// An iterator over the substrings of a string, separated by `sep`.
#[deriving(Clone)]
pub struct CharSplitIterator<'self,Sep> {
    priv string: &'self str,
    priv position: uint,
    priv sep: Sep,
    /// The number of splits remaining
    priv count: uint,
    /// Whether an empty string at the end is allowed
    priv allow_trailing_empty: bool,
    priv finished: bool,
    priv only_ascii: bool
}

/// An iterator over the words of a string, separated by an sequence of whitespace
pub type WordIterator<'self> =
    Filter<'self, &'self str, CharSplitIterator<'self, extern "Rust" fn(char) -> bool>>;

/// An iterator over the lines of a string, separated by either `\n` or (`\r\n`).
pub type AnyLineIterator<'self> =
    Map<'self, &'self str, &'self str, CharSplitIterator<'self, char>>;

impl<'self, Sep: CharEq> Iterator<&'self str> for CharSplitIterator<'self, Sep> {
    #[inline]
    fn next(&mut self) -> Option<&'self str> {
        if self.finished { return None }

        let l = self.string.len();
        let start = self.position;

        if self.only_ascii {
            // this gives a *huge* speed up for splitting on ASCII
            // characters (e.g. '\n' or ' ')
            while self.position < l && self.count > 0 {
                let byte = self.string[self.position];

                if self.sep.matches(byte as char) {
                    let slice = unsafe { raw::slice_bytes(self.string, start, self.position) };
                    self.position += 1;
                    self.count -= 1;
                    return Some(slice);
                }
                self.position += 1;
            }
        } else {
            while self.position < l && self.count > 0 {
                let CharRange {ch, next} = self.string.char_range_at(self.position);

                if self.sep.matches(ch) {
                    let slice = unsafe { raw::slice_bytes(self.string, start, self.position) };
                    self.position = next;
                    self.count -= 1;
                    return Some(slice);
                }
                self.position = next;
            }
        }
        self.finished = true;
        if self.allow_trailing_empty || start < l {
            Some(unsafe { raw::slice_bytes(self.string, start, l) })
        } else {
            None
        }
    }
}

/// An iterator over the start and end indices of the matches of a
/// substring within a larger string
#[deriving(Clone)]
pub struct MatchesIndexIterator<'self> {
    priv haystack: &'self str,
    priv needle: &'self str,
    priv position: uint,
}

/// An iterator over the substrings of a string separated by a given
/// search string
#[deriving(Clone)]
pub struct StrSplitIterator<'self> {
    priv it: MatchesIndexIterator<'self>,
    priv last_end: uint,
    priv finished: bool
}

impl<'self> Iterator<(uint, uint)> for MatchesIndexIterator<'self> {
    #[inline]
    fn next(&mut self) -> Option<(uint, uint)> {
        // See Issue #1932 for why this is a naive search
        let (h_len, n_len) = (self.haystack.len(), self.needle.len());
        let mut match_start = 0;
        let mut match_i = 0;

        while self.position < h_len {
            if self.haystack[self.position] == self.needle[match_i] {
                if match_i == 0 { match_start = self.position; }
                match_i += 1;
                self.position += 1;

                if match_i == n_len {
                    // found a match!
                    return Some((match_start, self.position));
                }
            } else {
                // failed match, backtrack
                if match_i > 0 {
                    match_i = 0;
                    self.position = match_start;
                }
                self.position += 1;
            }
        }
        None
    }
}

impl<'self> Iterator<&'self str> for StrSplitIterator<'self> {
    #[inline]
    fn next(&mut self) -> Option<&'self str> {
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

/// Replace all occurrences of one string with another
///
/// # Arguments
///
/// * s - The string containing substrings to replace
/// * from - The string to replace
/// * to - The replacement string
///
/// # Return value
///
/// The original string with all occurances of `from` replaced with `to`
pub fn replace(s: &str, from: &str, to: &str) -> ~str {
    let mut result = ~"";
    let mut last_end = 0;
    for (start, end) in s.matches_index_iter(from) {
        result.push_str(unsafe{raw::slice_bytes(s, last_end, start)});
        result.push_str(to);
        last_end = end;
    }
    result.push_str(unsafe{raw::slice_bytes(s, last_end, s.len())});
    result
}

/*
Section: Comparing strings
*/

/// Bytewise slice equality
#[cfg(not(test))]
#[lang="str_eq"]
#[inline]
pub fn eq_slice(a: &str, b: &str) -> bool {
    do a.as_imm_buf |ap, alen| {
        do b.as_imm_buf |bp, blen| {
            if (alen != blen) { false }
            else {
                unsafe {
                    libc::memcmp(ap as *libc::c_void,
                                 bp as *libc::c_void,
                                 alen as libc::size_t) == 0
                }
            }
        }
    }
}

/// Bytewise slice equality
#[cfg(test)]
#[inline]
pub fn eq_slice(a: &str, b: &str) -> bool {
    do a.as_imm_buf |ap, alen| {
        do b.as_imm_buf |bp, blen| {
            if (alen != blen) { false }
            else {
                unsafe {
                    libc::memcmp(ap as *libc::c_void,
                                 bp as *libc::c_void,
                                 alen as libc::size_t) == 0
                }
            }
        }
    }
}

/// Bytewise string equality
#[cfg(not(test))]
#[lang="uniq_str_eq"]
#[inline]
pub fn eq(a: &~str, b: &~str) -> bool {
    eq_slice(*a, *b)
}

#[cfg(test)]
#[inline]
pub fn eq(a: &~str, b: &~str) -> bool {
    eq_slice(*a, *b)
}

/*
Section: Searching
*/

// Utility used by various searching functions
fn match_at<'a,'b>(haystack: &'a str, needle: &'b str, at: uint) -> bool {
    let mut i = at;
    for c in needle.byte_iter() { if haystack[i] != c { return false; } i += 1u; }
    return true;
}

/*
Section: Misc
*/

/// Determines if a vector of bytes contains valid UTF-8
pub fn is_utf8(v: &[u8]) -> bool {
    let mut i = 0u;
    let total = v.len();
    fn unsafe_get(xs: &[u8], i: uint) -> u8 {
        unsafe { *xs.unsafe_ref(i) }
    }
    while i < total {
        let v_i = unsafe_get(v, i);
        if v_i < 128u8 {
            i += 1u;
        } else {
            let w = utf8_char_width(v_i);
            if w == 0u { return false; }

            let nexti = i + w;
            if nexti > total { return false; }

            // 2-byte encoding is for codepoints  \u0080 to  \u07ff
            //        first  C2 80        last DF BF
            // 3-byte encoding is for codepoints  \u0800 to  \uffff
            //        first  E0 A0 80     last EF BF BF
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
            // UTF8-tail   = %x80-BF
            // --
            // This code allows surrogate pairs: \uD800 to \uDFFF -> ED A0 80 to ED BF BF
            match w {
                2 => if unsafe_get(v, i + 1) & 192u8 != TAG_CONT_U8 {
                    return false
                },
                3 => match (v_i,
                            unsafe_get(v, i + 1),
                            unsafe_get(v, i + 2) & 192u8) {
                    (0xE0        , 0xA0 .. 0xBF, TAG_CONT_U8) => (),
                    (0xE1 .. 0xEF, 0x80 .. 0xBF, TAG_CONT_U8) => (),
                    _ => return false,
                },
                _ => match (v_i,
                            unsafe_get(v, i + 1),
                            unsafe_get(v, i + 2) & 192u8,
                            unsafe_get(v, i + 3) & 192u8) {
                    (0xF0        , 0x90 .. 0xBF, TAG_CONT_U8, TAG_CONT_U8) => (),
                    (0xF1 .. 0xF3, 0x80 .. 0xBF, TAG_CONT_U8, TAG_CONT_U8) => (),
                    (0xF4        , 0x80 .. 0x8F, TAG_CONT_U8, TAG_CONT_U8) => (),
                    _ => return false,
                },
            }

            i = nexti;
        }
    }
    true
}

/// Determines if a vector of `u16` contains valid UTF-16
pub fn is_utf16(v: &[u16]) -> bool {
    let len = v.len();
    let mut i = 0u;
    while (i < len) {
        let u = v[i];

        if  u <= 0xD7FF_u16 || u >= 0xE000_u16 {
            i += 1u;

        } else {
            if i+1u < len { return false; }
            let u2 = v[i+1u];
            if u < 0xD7FF_u16 || u > 0xDBFF_u16 { return false; }
            if u2 < 0xDC00_u16 || u2 > 0xDFFF_u16 { return false; }
            i += 2u;
        }
    }
    return true;
}

/// Iterates over the utf-16 characters in the specified slice, yielding each
/// decoded unicode character to the function provided.
///
/// # Failures
///
/// * Fails on invalid utf-16 data
pub fn utf16_chars(v: &[u16], f: &fn(char)) {
    let len = v.len();
    let mut i = 0u;
    while (i < len && v[i] != 0u16) {
        let u = v[i];

        if  u <= 0xD7FF_u16 || u >= 0xE000_u16 {
            f(u as char);
            i += 1u;

        } else {
            let u2 = v[i+1u];
            assert!(u >= 0xD800_u16 && u <= 0xDBFF_u16);
            assert!(u2 >= 0xDC00_u16 && u2 <= 0xDFFF_u16);
            let mut c = (u - 0xD800_u16) as char;
            c = c << 10;
            c |= (u2 - 0xDC00_u16) as char;
            c |= 0x1_0000_u32 as char;
            f(c);
            i += 2u;
        }
    }
}

/// Allocates a new string from the utf-16 slice provided
pub fn from_utf16(v: &[u16]) -> ~str {
    let mut buf = ~"";
    buf.reserve(v.len());
    utf16_chars(v, |ch| buf.push_char(ch));
    buf
}

/// Allocates a new string with the specified capacity. The string returned is
/// the empty string, but has capacity for much more.
#[inline]
pub fn with_capacity(capacity: uint) -> ~str {
    unsafe {
        cast::transmute(vec::with_capacity::<~[u8]>(capacity))
    }
}

/// As char_len but for a slice of a string
///
/// # Arguments
///
/// * s - A valid string
/// * start - The position inside `s` where to start counting in bytes
/// * end - The position where to stop counting
///
/// # Return value
///
/// The number of Unicode characters in `s` between the given indices.
pub fn count_chars(s: &str, start: uint, end: uint) -> uint {
    assert!(s.is_char_boundary(start));
    assert!(s.is_char_boundary(end));
    let mut i = start;
    let mut len = 0u;
    while i < end {
        let next = s.char_range_at(i).next;
        len += 1u;
        i = next;
    }
    return len;
}

/// Counts the number of bytes taken by the first `n` chars in `s`
/// starting from `start`.
pub fn count_bytes<'b>(s: &'b str, start: uint, n: uint) -> uint {
    assert!(s.is_char_boundary(start));
    let mut end = start;
    let mut cnt = n;
    let l = s.len();
    while cnt > 0u {
        assert!(end < l);
        let next = s.char_range_at(end).next;
        cnt -= 1u;
        end = next;
    }
    end - start
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
pub fn utf8_char_width(b: u8) -> uint {
    return UTF8_CHAR_WIDTH[b] as uint;
}

#[allow(missing_doc)]
pub struct CharRange {
    ch: char,
    next: uint
}

// Return the initial codepoint accumulator for the first byte.
// The first byte is special, only want bottom 5 bits for width 2, 4 bits
// for width 3, and 3 bits for width 4
macro_rules! utf8_first_byte(
    ($byte:expr, $width:expr) => (($byte & (0x7F >> $width)) as uint)
)

// return the value of $ch updated with continuation byte $byte
macro_rules! utf8_acc_cont_byte(
    ($ch:expr, $byte:expr) => (($ch << 6) | ($byte & 63u8) as uint)
)

static TAG_CONT_U8: u8 = 128u8;
static MAX_UNICODE: uint = 1114112u;

/// Unsafe operations
pub mod raw {
    use option::Some;
    use cast;
    use libc;
    use ptr;
    use str::is_utf8;
    use vec;
    use vec::MutableVector;
    use unstable::raw::Slice;

    /// Create a Rust string from a *u8 buffer of the given length
    pub unsafe fn from_buf_len(buf: *u8, len: uint) -> ~str {
        let mut v: ~[u8] = vec::with_capacity(len);
        do v.as_mut_buf |vbuf, _len| {
            ptr::copy_memory(vbuf, buf as *u8, len)
        };
        vec::raw::set_len(&mut v, len);

        assert!(is_utf8(v));
        ::cast::transmute(v)
    }

    #[lang="strdup_uniq"]
    #[cfg(not(test))]
    #[allow(missing_doc)]
    #[inline]
    pub unsafe fn strdup_uniq(ptr: *u8, len: uint) -> ~str {
        from_buf_len(ptr, len)
    }

    /// Create a Rust string from a null-terminated C string
    pub unsafe fn from_c_str(buf: *libc::c_char) -> ~str {
        let mut curr = buf;
        let mut i = 0;
        while *curr != 0 {
            i += 1;
            curr = ptr::offset(buf, i);
        }
        from_buf_len(buf as *u8, i as uint)
    }

    /// Converts a vector of bytes to a new owned string.
    pub unsafe fn from_bytes(v: &[u8]) -> ~str {
        do v.as_imm_buf |buf, len| {
            from_buf_len(buf, len)
        }
    }

    /// Converts an owned vector of bytes to a new owned string. This assumes
    /// that the utf-8-ness of the vector has already been validated
    #[inline]
    pub unsafe fn from_bytes_owned(v: ~[u8]) -> ~str {
        cast::transmute(v)
    }

    /// Converts a byte to a string.
    pub unsafe fn from_byte(u: u8) -> ~str { from_bytes([u]) }

    /// Form a slice from a C string. Unsafe because the caller must ensure the
    /// C string has the static lifetime, or else the return value may be
    /// invalidated later.
    pub unsafe fn c_str_to_static_slice(s: *libc::c_char) -> &'static str {
        let s = s as *u8;
        let mut curr = s;
        let mut len = 0u;
        while *curr != 0u8 {
            len += 1u;
            curr = ptr::offset(s, len as int);
        }
        let v = Slice { data: s, len: len };
        assert!(is_utf8(::cast::transmute(v)));
        ::cast::transmute(v)
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
        do s.as_imm_buf |sbuf, n| {
             assert!((begin <= end));
             assert!((end <= n));

             cast::transmute(Slice {
                 data: ptr::offset(sbuf, begin as int),
                 len: end - begin,
             })
        }
    }

    /// Appends a byte to a string. (Not UTF-8 safe).
    #[inline]
    pub unsafe fn push_byte(s: &mut ~str, b: u8) {
        let v: &mut ~[u8] = cast::transmute(s);
        v.push(b);
    }

    /// Appends a vector of bytes to a string. (Not UTF-8 safe).
    unsafe fn push_bytes(s: &mut ~str, bytes: &[u8]) {
        let new_len = s.len() + bytes.len();
        s.reserve_at_least(new_len);
        for byte in bytes.iter() { push_byte(&mut *s, *byte); }
    }

    /// Removes the last byte from a string and returns it. (Not UTF-8 safe).
    pub unsafe fn pop_byte(s: &mut ~str) -> u8 {
        let len = s.len();
        assert!((len > 0u));
        let b = s[len - 1u];
        set_len(s, len - 1u);
        return b;
    }

    /// Removes the first byte from a string and returns it. (Not UTF-8 safe).
    pub unsafe fn shift_byte(s: &mut ~str) -> u8 {
        let len = s.len();
        assert!((len > 0u));
        let b = s[0];
        *s = s.slice(1, len).to_owned();
        return b;
    }

    /// Sets the length of a string
    ///
    /// This will explicitly set the size of the string, without actually
    /// modifying its buffers, so it is up to the caller to ensure that
    /// the string is actually the specified size.
    #[inline]
    pub unsafe fn set_len(s: &mut ~str, new_len: uint) {
        let v: &mut ~[u8] = cast::transmute(s);
        vec::raw::set_len(v, new_len)
    }

    /// Sets the length of a string
    ///
    /// This will explicitly set the size of the string, without actually
    /// modifing its buffers, so it is up to the caller to ensure that
    /// the string is actually the specified size.
    #[test]
    fn test_from_buf_len() {
        unsafe {
            let a = ~[65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 0u8];
            let b = vec::raw::to_ptr(a);
            let c = from_buf_len(b, 3u);
            assert_eq!(c, ~"AAA");
        }
    }

}

/*
Section: Trait implementations
*/

#[cfg(not(test))]
pub mod traits {
    use ops::Add;
    use cmp::{TotalOrd, Ordering, Less, Equal, Greater, Eq, Ord, Equiv, TotalEq};
    use super::{Str, eq_slice};
    use option::{Some, None};

    impl<'self> Add<&'self str,~str> for &'self str {
        #[inline]
        fn add(&self, rhs: & &'self str) -> ~str {
            let mut ret = self.to_owned();
            ret.push_str(*rhs);
            ret
        }
    }

    impl<'self> TotalOrd for &'self str {
        #[inline]
        fn cmp(&self, other: & &'self str) -> Ordering {
            for (s_b, o_b) in self.byte_iter().zip(other.byte_iter()) {
                match s_b.cmp(&o_b) {
                    Greater => return Greater,
                    Less => return Less,
                    Equal => ()
                }
            }

            self.len().cmp(&other.len())
        }
    }

    impl TotalOrd for ~str {
        #[inline]
        fn cmp(&self, other: &~str) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl TotalOrd for @str {
        #[inline]
        fn cmp(&self, other: &@str) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl<'self> Eq for &'self str {
        #[inline]
        fn eq(&self, other: & &'self str) -> bool {
            eq_slice((*self), (*other))
        }
        #[inline]
        fn ne(&self, other: & &'self str) -> bool { !(*self).eq(other) }
    }

    impl Eq for ~str {
        #[inline]
        fn eq(&self, other: &~str) -> bool {
            eq_slice((*self), (*other))
        }
        #[inline]
        fn ne(&self, other: &~str) -> bool { !(*self).eq(other) }
    }

    impl Eq for @str {
        #[inline]
        fn eq(&self, other: &@str) -> bool {
            eq_slice((*self), (*other))
        }
        #[inline]
        fn ne(&self, other: &@str) -> bool { !(*self).eq(other) }
    }

    impl<'self> TotalEq for &'self str {
        #[inline]
        fn equals(&self, other: & &'self str) -> bool {
            eq_slice((*self), (*other))
        }
    }

    impl TotalEq for ~str {
        #[inline]
        fn equals(&self, other: &~str) -> bool {
            eq_slice((*self), (*other))
        }
    }

    impl TotalEq for @str {
        #[inline]
        fn equals(&self, other: &@str) -> bool {
            eq_slice((*self), (*other))
        }
    }

    impl<'self> Ord for &'self str {
        #[inline]
        fn lt(&self, other: & &'self str) -> bool { self.cmp(other) == Less }
    }

    impl Ord for ~str {
        #[inline]
        fn lt(&self, other: &~str) -> bool { self.cmp(other) == Less }
    }

    impl Ord for @str {
        #[inline]
        fn lt(&self, other: &@str) -> bool { self.cmp(other) == Less }
    }

    impl<'self, S: Str> Equiv<S> for &'self str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(*self, other.as_slice()) }
    }

    impl<'self, S: Str> Equiv<S> for @str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(*self, other.as_slice()) }
    }

    impl<'self, S: Str> Equiv<S> for ~str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(*self, other.as_slice()) }
    }
}

#[cfg(test)]
pub mod traits {}

/// Any string that can be represented as a slice
pub trait Str {
    /// Work with `self` as a slice.
    fn as_slice<'a>(&'a self) -> &'a str;

    /// Convert `self` into a ~str, not making a copy if possible
    fn into_owned(self) -> ~str;
}

impl<'self> Str for &'self str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str { *self }

    #[inline]
    fn into_owned(self) -> ~str { self.to_owned() }
}

impl<'self> Str for ~str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str {
        let s: &'a str = *self; s
    }

    #[inline]
    fn into_owned(self) -> ~str { self }
}

impl<'self> Str for @str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str {
        let s: &'a str = *self; s
    }

    #[inline]
    fn into_owned(self) -> ~str { self.to_owned() }
}

impl<'self> Container for &'self str {
    #[inline]
    fn len(&self) -> uint {
        do self.as_imm_buf |_p, n| { n }
    }
}

impl Container for ~str {
    #[inline]
    fn len(&self) -> uint { self.as_slice().len() }
}

impl Container for @str {
    #[inline]
    fn len(&self) -> uint { self.as_slice().len() }
}

impl Mutable for ~str {
    /// Remove all content, make the string empty
    #[inline]
    fn clear(&mut self) {
        unsafe {
            raw::set_len(self, 0)
        }
    }
}

#[allow(missing_doc)]
pub trait StrSlice<'self> {
    fn contains<'a>(&self, needle: &'a str) -> bool;
    fn contains_char(&self, needle: char) -> bool;
    fn iter(&self) -> CharIterator<'self>;
    fn rev_iter(&self) -> CharRevIterator<'self>;
    fn byte_iter(&self) -> ByteIterator<'self>;
    fn byte_rev_iter(&self) -> ByteRevIterator<'self>;
    fn char_offset_iter(&self) -> CharOffsetIterator<'self>;
    fn char_offset_rev_iter(&self) -> CharOffsetRevIterator<'self>;
    fn split_iter<Sep: CharEq>(&self, sep: Sep) -> CharSplitIterator<'self, Sep>;
    fn splitn_iter<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitIterator<'self, Sep>;
    fn split_options_iter<Sep: CharEq>(&self, sep: Sep, count: uint, allow_trailing_empty: bool)
        -> CharSplitIterator<'self, Sep>;
    fn matches_index_iter(&self, sep: &'self str) -> MatchesIndexIterator<'self>;
    fn split_str_iter(&self, &'self str) -> StrSplitIterator<'self>;
    fn line_iter(&self) -> CharSplitIterator<'self, char>;
    fn any_line_iter(&self) -> AnyLineIterator<'self>;
    fn word_iter(&self) -> WordIterator<'self>;
    fn ends_with(&self, needle: &str) -> bool;
    fn is_whitespace(&self) -> bool;
    fn is_alphanumeric(&self) -> bool;
    fn char_len(&self) -> uint;

    fn slice(&self, begin: uint, end: uint) -> &'self str;
    fn slice_from(&self, begin: uint) -> &'self str;
    fn slice_to(&self, end: uint) -> &'self str;

    fn slice_chars(&self, begin: uint, end: uint) -> &'self str;

    fn starts_with(&self, needle: &str) -> bool;
    fn escape_default(&self) -> ~str;
    fn escape_unicode(&self) -> ~str;
    fn trim(&self) -> &'self str;
    fn trim_left(&self) -> &'self str;
    fn trim_right(&self) -> &'self str;
    fn trim_chars<C: CharEq>(&self, to_trim: &C) -> &'self str;
    fn trim_left_chars<C: CharEq>(&self, to_trim: &C) -> &'self str;
    fn trim_right_chars<C: CharEq>(&self, to_trim: &C) -> &'self str;
    fn replace(&self, from: &str, to: &str) -> ~str;
    fn to_owned(&self) -> ~str;
    fn to_managed(&self) -> @str;
    fn to_utf16(&self) -> ~[u16];
    fn is_char_boundary(&self, index: uint) -> bool;
    fn char_range_at(&self, start: uint) -> CharRange;
    fn char_at(&self, i: uint) -> char;
    fn char_range_at_reverse(&self, start: uint) -> CharRange;
    fn char_at_reverse(&self, i: uint) -> char;
    fn as_bytes(&self) -> &'self [u8];

    fn find<C: CharEq>(&self, search: C) -> Option<uint>;
    fn rfind<C: CharEq>(&self, search: C) -> Option<uint>;
    fn find_str(&self, &str) -> Option<uint>;

    fn repeat(&self, nn: uint) -> ~str;

    fn slice_shift_char(&self) -> (char, &'self str);

    fn map_chars(&self, ff: &fn(char) -> char) -> ~str;

    fn lev_distance(&self, t: &str) -> uint;

    fn subslice_offset(&self, inner: &str) -> uint;

    fn as_imm_buf<T>(&self, f: &fn(*u8, uint) -> T) -> T;
}

/// Extension methods for strings
impl<'self> StrSlice<'self> for &'self str {
    /// Returns true if one string contains another
    ///
    /// # Arguments
    ///
    /// * needle - The string to look for
    #[inline]
    fn contains<'a>(&self, needle: &'a str) -> bool {
        self.find_str(needle).is_some()
    }

    /// Returns true if a string contains a char.
    ///
    /// # Arguments
    ///
    /// * needle - The char to look for
    #[inline]
    fn contains_char(&self, needle: char) -> bool {
        self.find(needle).is_some()
    }

    /// An iterator over the characters of `self`. Note, this iterates
    /// over unicode code-points, not unicode graphemes.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let v: ~[char] = "abc åäö".iter().collect();
    /// assert_eq!(v, ~['a', 'b', 'c', ' ', 'å', 'ä', 'ö']);
    /// ~~~
    #[inline]
    fn iter(&self) -> CharIterator<'self> {
        self.char_offset_iter().map(|(_, c)| c)
    }

    /// An iterator over the characters of `self`, in reverse order.
    #[inline]
    fn rev_iter(&self) -> CharRevIterator<'self> {
        self.iter().invert()
    }

    /// An iterator over the bytes of `self`
    #[inline]
    fn byte_iter(&self) -> ByteIterator<'self> {
        self.as_bytes().iter().map(|&b| b)
    }

    /// An iterator over the bytes of `self`, in reverse order
    #[inline]
    fn byte_rev_iter(&self) -> ByteRevIterator<'self> {
        self.byte_iter().invert()
    }

    /// An iterator over the characters of `self` and their byte offsets.
    #[inline]
    fn char_offset_iter(&self) -> CharOffsetIterator<'self> {
        CharOffsetIterator {
            index_front: 0,
            index_back: self.len(),
            string: *self
        }
    }

    /// An iterator over the characters of `self` and their byte offsets.
    #[inline]
    fn char_offset_rev_iter(&self) -> CharOffsetRevIterator<'self> {
        self.char_offset_iter().invert()
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let v: ~[&str] = "Mary had a little lamb".split_iter(' ').collect();
    /// assert_eq!(v, ~["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: ~[&str] = "abc1def2ghi".split_iter(|c: char| c.is_digit()).collect();
    /// assert_eq!(v, ~["abc", "def", "ghi"]);
    /// ~~~
    #[inline]
    fn split_iter<Sep: CharEq>(&self, sep: Sep) -> CharSplitIterator<'self, Sep> {
        self.split_options_iter(sep, self.len(), true)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, restricted to splitting at most `count`
    /// times.
    #[inline]
    fn splitn_iter<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitIterator<'self, Sep> {
        self.split_options_iter(sep, count, true)
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, splitting at most `count` times, and
    /// possibly not including the trailing empty substring, if it
    /// exists.
    #[inline]
    fn split_options_iter<Sep: CharEq>(&self, sep: Sep, count: uint, allow_trailing_empty: bool)
        -> CharSplitIterator<'self, Sep> {
        let only_ascii = sep.only_ascii();
        CharSplitIterator {
            string: *self,
            position: 0,
            sep: sep,
            count: count,
            allow_trailing_empty: allow_trailing_empty,
            finished: false,
            only_ascii: only_ascii
        }
    }

    /// An iterator over the start and end indices of each match of
    /// `sep` within `self`.
    #[inline]
    fn matches_index_iter(&self, sep: &'self str) -> MatchesIndexIterator<'self> {
        assert!(!sep.is_empty())
        MatchesIndexIterator {
            haystack: *self,
            needle: sep,
            position: 0
        }
    }

    /// An iterator over the substrings of `self` separated by `sep`.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let v: ~[&str] = "abcXXXabcYYYabc".split_str_iter("abc").collect()
    /// assert_eq!(v, ["", "XXX", "YYY", ""]);
    /// ~~~
    #[inline]
    fn split_str_iter(&self, sep: &'self str) -> StrSplitIterator<'self> {
        StrSplitIterator {
            it: self.matches_index_iter(sep),
            last_end: 0,
            finished: false
        }
    }

    /// An iterator over the lines of a string (subsequences separated
    /// by `\n`).
    #[inline]
    fn line_iter(&self) -> CharSplitIterator<'self, char> {
        self.split_options_iter('\n', self.len(), false)
    }

    /// An iterator over the lines of a string, separated by either
    /// `\n` or (`\r\n`).
    fn any_line_iter(&self) -> AnyLineIterator<'self> {
        do self.line_iter().map |line| {
            let l = line.len();
            if l > 0 && line[l - 1] == '\r' as u8 { line.slice(0, l - 1) }
            else { line }
        }
    }

    /// An iterator over the words of a string (subsequences separated
    /// by any sequence of whitespace).
    #[inline]
    fn word_iter(&self) -> WordIterator<'self> {
        self.split_iter(char::is_whitespace).filter(|s| !s.is_empty())
    }

    /// Returns true if the string contains only whitespace
    ///
    /// Whitespace characters are determined by `char::is_whitespace`
    #[inline]
    fn is_whitespace(&self) -> bool { self.iter().all(char::is_whitespace) }

    /// Returns true if the string contains only alphanumerics
    ///
    /// Alphanumeric characters are determined by `char::is_alphanumeric`
    #[inline]
    fn is_alphanumeric(&self) -> bool { self.iter().all(char::is_alphanumeric) }

    /// Returns the number of characters that a string holds
    #[inline]
    fn char_len(&self) -> uint { self.iter().len() }

    /// Returns a slice of the given string from the byte range
    /// [`begin`..`end`)
    ///
    /// Fails when `begin` and `end` do not point to valid characters or
    /// beyond the last character of the string
    #[inline]
    fn slice(&self, begin: uint, end: uint) -> &'self str {
        assert!(self.is_char_boundary(begin));
        assert!(self.is_char_boundary(end));
        unsafe { raw::slice_bytes(*self, begin, end) }
    }

    /// Returns a slice of the string from `begin` to its end.
    ///
    /// Fails when `begin` does not point to a valid character, or is
    /// out of bounds.
    #[inline]
    fn slice_from(&self, begin: uint) -> &'self str {
        self.slice(begin, self.len())
    }

    /// Returns a slice of the string from the beginning to byte
    /// `end`.
    ///
    /// Fails when `end` does not point to a valid character, or is
    /// out of bounds.
    #[inline]
    fn slice_to(&self, end: uint) -> &'self str {
        self.slice(0, end)
    }

    /// Returns a slice of the string from the char range
    /// [`begin`..`end`).
    ///
    /// Fails if `begin` > `end` or the either `begin` or `end` are
    /// beyond the last character of the string.
    fn slice_chars(&self, begin: uint, end: uint) -> &'self str {
        assert!(begin <= end);
        // not sure how to use the iterators for this nicely.
        let mut position = 0;
        let mut count = 0;
        let l = self.len();
        while count < begin && position < l {
            position = self.char_range_at(position).next;
            count += 1;
        }
        if count < begin { fail!("Attempted to begin slice_chars beyond end of string") }
        let start_byte = position;
        while count < end && position < l {
            position = self.char_range_at(position).next;
            count += 1;
        }
        if count < end { fail!("Attempted to end slice_chars beyond end of string") }

        self.slice(start_byte, position)
    }

    /// Returns true if `needle` is a prefix of the string.
    fn starts_with<'a>(&self, needle: &'a str) -> bool {
        let (self_len, needle_len) = (self.len(), needle.len());
        if needle_len == 0u { true }
        else if needle_len > self_len { false }
        else { match_at(*self, needle, 0u) }
    }

    /// Returns true if `needle` is a suffix of the string.
    fn ends_with(&self, needle: &str) -> bool {
        let (self_len, needle_len) = (self.len(), needle.len());
        if needle_len == 0u { true }
        else if needle_len > self_len { false }
        else { match_at(*self, needle, self_len - needle_len) }
    }

    /// Escape each char in `s` with char::escape_default.
    fn escape_default(&self) -> ~str {
        let mut out: ~str = ~"";
        out.reserve_at_least(self.len());
        for c in self.iter() {
            do c.escape_default |c| {
                out.push_char(c);
            }
        }
        out
    }

    /// Escape each char in `s` with char::escape_unicode.
    fn escape_unicode(&self) -> ~str {
        let mut out: ~str = ~"";
        out.reserve_at_least(self.len());
        for c in self.iter() {
            do c.escape_unicode |c| {
                out.push_char(c);
            }
        }
        out
    }

    /// Returns a string with leading and trailing whitespace removed
    #[inline]
    fn trim(&self) -> &'self str {
        self.trim_left().trim_right()
    }

    /// Returns a string with leading whitespace removed
    #[inline]
    fn trim_left(&self) -> &'self str {
        self.trim_left_chars(&char::is_whitespace)
    }

    /// Returns a string with trailing whitespace removed
    #[inline]
    fn trim_right(&self) -> &'self str {
        self.trim_right_chars(&char::is_whitespace)
    }

    /// Returns a string with characters that match `to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// assert_eq!("11foo1bar11".trim_chars(&'1'), "foo1bar")
    /// assert_eq!("12foo1bar12".trim_chars(& &['1', '2']), "foo1bar")
    /// assert_eq!("123foo1bar123".trim_chars(&|c: char| c.is_digit()), "foo1bar")
    /// ~~~
    #[inline]
    fn trim_chars<C: CharEq>(&self, to_trim: &C) -> &'self str {
        self.trim_left_chars(to_trim).trim_right_chars(to_trim)
    }

    /// Returns a string with leading `chars_to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// assert_eq!("11foo1bar11".trim_left_chars(&'1'), "foo1bar11")
    /// assert_eq!("12foo1bar12".trim_left_chars(& &['1', '2']), "foo1bar12")
    /// assert_eq!("123foo1bar123".trim_left_chars(&|c: char| c.is_digit()), "foo1bar123")
    /// ~~~
    #[inline]
    fn trim_left_chars<C: CharEq>(&self, to_trim: &C) -> &'self str {
        match self.find(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(first) => unsafe { raw::slice_bytes(*self, first, self.len()) }
        }
    }

    /// Returns a string with trailing `chars_to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// assert_eq!("11foo1bar11".trim_right_chars(&'1'), "11foo1bar")
    /// assert_eq!("12foo1bar12".trim_right_chars(& &['1', '2']), "12foo1bar")
    /// assert_eq!("123foo1bar123".trim_right_chars(&|c: char| c.is_digit()), "123foo1bar")
    /// ~~~
    #[inline]
    fn trim_right_chars<C: CharEq>(&self, to_trim: &C) -> &'self str {
        match self.rfind(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(last) => {
                let next = self.char_range_at(last).next;
                unsafe { raw::slice_bytes(*self, 0u, next) }
            }
        }
    }

    /// Replace all occurrences of one string with another
    ///
    /// # Arguments
    ///
    /// * from - The string to replace
    /// * to - The replacement string
    ///
    /// # Return value
    ///
    /// The original string with all occurances of `from` replaced with `to`
    fn replace(&self, from: &str, to: &str) -> ~str {
        let mut result = ~"";
        let mut last_end = 0;
        for (start, end) in self.matches_index_iter(from) {
            result.push_str(unsafe{raw::slice_bytes(*self, last_end, start)});
            result.push_str(to);
            last_end = end;
        }
        result.push_str(unsafe{raw::slice_bytes(*self, last_end, self.len())});
        result
    }

    /// Copy a slice into a new unique str
    #[inline]
    fn to_owned(&self) -> ~str {
        do self.as_imm_buf |src, len| {
            unsafe {
                let mut v = vec::with_capacity(len);

                do v.as_mut_buf |dst, _| {
                    ptr::copy_memory(dst, src, len);
                }
                vec::raw::set_len(&mut v, len);
                ::cast::transmute(v)
            }
        }
    }

    #[inline]
    fn to_managed(&self) -> @str {
        unsafe {
            let v: *&[u8] = cast::transmute(self);
            cast::transmute(at_vec::to_managed(*v))
        }
    }

    /// Converts to a vector of `u16` encoded as UTF-16.
    fn to_utf16(&self) -> ~[u16] {
        let mut u = ~[];
        for ch in self.iter() {
            // Arithmetic with u32 literals is easier on the eyes than chars.
            let mut ch = ch as u32;

            if (ch & 0xFFFF_u32) == ch {
                // The BMP falls through (assuming non-surrogate, as it
                // should)
                assert!(ch <= 0xD7FF_u32 || ch >= 0xE000_u32);
                u.push(ch as u16)
            } else {
                // Supplementary planes break into surrogates.
                assert!(ch >= 0x1_0000_u32 && ch <= 0x10_FFFF_u32);
                ch -= 0x1_0000_u32;
                let w1 = 0xD800_u16 | ((ch >> 10) as u16);
                let w2 = 0xDC00_u16 | ((ch as u16) & 0x3FF_u16);
                u.push_all([w1, w2])
            }
        }
        u
    }

    /// Returns false if the index points into the middle of a multi-byte
    /// character sequence.
    fn is_char_boundary(&self, index: uint) -> bool {
        if index == self.len() { return true; }
        let b = self[index];
        return b < 128u8 || b >= 192u8;
    }

    /// Pluck a character out of a string and return the index of the next
    /// character.
    ///
    /// This function can be used to iterate over the unicode characters of a
    /// string.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let s = "中华Việt Nam";
    /// let i = 0u;
    /// while i < s.len() {
    ///     let CharRange {ch, next} = s.char_range_at(i);
    ///     printfln!("%u: %c", i, ch);
    ///     i = next;
    /// }
    /// ~~~
    ///
    /// # Example output
    ///
    /// ~~~
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
    /// ~~~
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
    #[inline]
    fn char_range_at(&self, i: uint) -> CharRange {
        if (self[i] < 128u8) {
            return CharRange {ch: self[i] as char, next: i + 1 };
        }

        // Multibyte case is a fn to allow char_range_at to inline cleanly
        fn multibyte_char_range_at(s: &str, i: uint) -> CharRange {
            let mut val = s[i] as uint;
            let w = UTF8_CHAR_WIDTH[val] as uint;
            assert!((w != 0));

            val = utf8_first_byte!(val, w);
            val = utf8_acc_cont_byte!(val, s[i + 1]);
            if w > 2 { val = utf8_acc_cont_byte!(val, s[i + 2]); }
            if w > 3 { val = utf8_acc_cont_byte!(val, s[i + 3]); }

            return CharRange {ch: val as char, next: i + w};
        }

        return multibyte_char_range_at(*self, i);
    }

    /// Plucks the character starting at the `i`th byte of a string
    #[inline]
    fn char_at(&self, i: uint) -> char { self.char_range_at(i).ch }

    /// Given a byte position and a str, return the previous char and its position.
    ///
    /// This function can be used to iterate over a unicode string in reverse.
    ///
    /// Returns 0 for next index if called on start index 0.
    fn char_range_at_reverse(&self, start: uint) -> CharRange {
        let mut prev = start;

        // while there is a previous byte == 10......
        while prev > 0u && self[prev - 1u] & 192u8 == TAG_CONT_U8 {
            prev -= 1u;
        }

        // now refer to the initial byte of previous char
        if prev > 0u {
            prev -= 1u;
        } else {
            prev = 0u;
        }


        let ch = self.char_at(prev);
        return CharRange {ch:ch, next:prev};
    }

    /// Plucks the character ending at the `i`th byte of a string
    #[inline]
    fn char_at_reverse(&self, i: uint) -> char {
        self.char_range_at_reverse(i).ch
    }

    /// Work with the byte buffer of a string as a byte slice.
    ///
    /// The byte slice does not include the null terminator.
    fn as_bytes(&self) -> &'self [u8] {
        unsafe { cast::transmute(*self) }
    }

    /// Returns the byte index of the first character of `self` that matches `search`
    ///
    /// # Return value
    ///
    /// `Some` containing the byte index of the last matching character
    /// or `None` if there is no match
    fn find<C: CharEq>(&self, search: C) -> Option<uint> {
        if search.only_ascii() {
            for (i, b) in self.byte_iter().enumerate() {
                if search.matches(b as char) { return Some(i) }
            }
        } else {
            let mut index = 0;
            for c in self.iter() {
                if search.matches(c) { return Some(index); }
                index += c.len_utf8_bytes();
            }
        }

        None
    }

    /// Returns the byte index of the last character of `self` that matches `search`
    ///
    /// # Return value
    ///
    /// `Some` containing the byte index of the last matching character
    /// or `None` if there is no match
    fn rfind<C: CharEq>(&self, search: C) -> Option<uint> {
        let mut index = self.len();
        if search.only_ascii() {
            for b in self.byte_rev_iter() {
                index -= 1;
                if search.matches(b as char) { return Some(index); }
            }
        } else {
            for c in self.rev_iter() {
                index -= c.len_utf8_bytes();
                if search.matches(c) { return Some(index); }
            }
        }

        None
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
    /// or `None` if there is no match
    fn find_str(&self, needle: &str) -> Option<uint> {
        if needle.is_empty() {
            Some(0)
        } else {
            self.matches_index_iter(needle)
                .next()
                .map_move(|(start, _end)| start)
        }
    }

    /// Given a string, make a new string with repeated copies of it.
    fn repeat(&self, nn: uint) -> ~str {
        do self.as_imm_buf |buf, len| {
            let mut ret = with_capacity(nn * len);

            unsafe {
                do ret.as_mut_buf |rbuf, _len| {
                    let mut rbuf = rbuf;

                    do nn.times {
                        ptr::copy_memory(rbuf, buf, len);
                        rbuf = rbuf.offset(len as int);
                    }
                }
                raw::set_len(&mut ret, nn * len);
            }
            ret
        }
    }

    /// Retrieves the first character from a string slice and returns
    /// it. This does not allocate a new string; instead, it returns a
    /// slice that point one character beyond the character that was
    /// shifted.
    ///
    /// # Failure
    ///
    /// If the string does not contain any characters
    #[inline]
    fn slice_shift_char(&self) -> (char, &'self str) {
        let CharRange {ch, next} = self.char_range_at(0u);
        let next_s = unsafe { raw::slice_bytes(*self, next, self.len()) };
        return (ch, next_s);
    }

    /// Apply a function to each character.
    fn map_chars(&self, ff: &fn(char) -> char) -> ~str {
        let mut result = with_capacity(self.len());
        for cc in self.iter() {
            result.push_char(ff(cc));
        }
        result
    }

    /// Levenshtein Distance between two strings.
    fn lev_distance(&self, t: &str) -> uint {
        let slen = self.len();
        let tlen = t.len();

        if slen == 0 { return tlen; }
        if tlen == 0 { return slen; }

        let mut dcol = vec::from_fn(tlen + 1, |x| x);

        for (i, sc) in self.iter().enumerate() {

            let mut current = i;
            dcol[0] = current + 1;

            for (j, tc) in t.iter().enumerate() {

                let next = dcol[j + 1];

                if sc == tc {
                    dcol[j + 1] = current;
                } else {
                    dcol[j + 1] = ::cmp::min(current, next);
                    dcol[j + 1] = ::cmp::min(dcol[j + 1], dcol[j]) + 1;
                }

                current = next;
            }
        }

        return dcol[tlen];
    }

    /// Returns the byte offset of an inner slice relative to an enclosing outer slice.
    ///
    /// Fails if `inner` is not a direct slice contained within self.
    ///
    /// # Example
    ///
    /// ~~~ {.rust}
    /// let string = "a\nb\nc";
    /// let mut lines = ~[];
    /// for line in string.line_iter() { lines.push(line) }
    ///
    /// assert!(string.subslice_offset(lines[0]) == 0); // &"a"
    /// assert!(string.subslice_offset(lines[1]) == 2); // &"b"
    /// assert!(string.subslice_offset(lines[2]) == 4); // &"c"
    /// ~~~
    #[inline]
    fn subslice_offset(&self, inner: &str) -> uint {
        do self.as_imm_buf |a, a_len| {
            do inner.as_imm_buf |b, b_len| {
                let a_start: uint;
                let a_end: uint;
                let b_start: uint;
                let b_end: uint;
                unsafe {
                    a_start = cast::transmute(a); a_end = a_len + cast::transmute(a);
                    b_start = cast::transmute(b); b_end = b_len + cast::transmute(b);
                }
                assert!(a_start <= b_start);
                assert!(b_end <= a_end);
                b_start - a_start
            }
        }
    }

    /// Work with the byte buffer and length of a slice.
    ///
    /// The given length is one byte longer than the 'official' indexable
    /// length of the string. This is to permit probing the byte past the
    /// indexable area for a null byte, as is the case in slices pointing
    /// to full strings, or suffixes of them.
    #[inline]
    fn as_imm_buf<T>(&self, f: &fn(*u8, uint) -> T) -> T {
        let v: &[u8] = unsafe { cast::transmute(*self) };
        v.as_imm_buf(f)
    }
}

#[allow(missing_doc)]
pub trait OwnedStr {
    fn push_str_no_overallocate(&mut self, rhs: &str);
    fn push_str(&mut self, rhs: &str);
    fn push_char(&mut self, c: char);
    fn pop_char(&mut self) -> char;
    fn shift_char(&mut self) -> char;
    fn unshift_char(&mut self, ch: char);
    fn append(self, rhs: &str) -> ~str;
    fn reserve(&mut self, n: uint);
    fn reserve_at_least(&mut self, n: uint);
    fn capacity(&self) -> uint;

    /// Work with the mutable byte buffer and length of a slice.
    ///
    /// The given length is one byte longer than the 'official' indexable
    /// length of the string. This is to permit probing the byte past the
    /// indexable area for a null byte, as is the case in slices pointing
    /// to full strings, or suffixes of them.
    ///
    /// Make sure any mutations to this buffer keep this string valid UTF8.
    fn as_mut_buf<T>(&mut self, f: &fn(*mut u8, uint) -> T) -> T;
}

impl OwnedStr for ~str {
    /// Appends a string slice to the back of a string, without overallocating
    #[inline]
    fn push_str_no_overallocate(&mut self, rhs: &str) {
        unsafe {
            let llen = self.len();
            let rlen = rhs.len();
            self.reserve(llen + rlen);
            do self.as_imm_buf |lbuf, _llen| {
                do rhs.as_imm_buf |rbuf, _rlen| {
                    let dst = ptr::offset(lbuf, llen as int);
                    let dst = cast::transmute_mut_unsafe(dst);
                    ptr::copy_memory(dst, rbuf, rlen);
                }
            }
            raw::set_len(self, llen + rlen);
        }
    }

    /// Appends a string slice to the back of a string
    #[inline]
    fn push_str(&mut self, rhs: &str) {
        unsafe {
            let llen = self.len();
            let rlen = rhs.len();
            self.reserve_at_least(llen + rlen);
            do self.as_imm_buf |lbuf, _llen| {
                do rhs.as_imm_buf |rbuf, _rlen| {
                    let dst = ptr::offset(lbuf, llen as int);
                    let dst = cast::transmute_mut_unsafe(dst);
                    ptr::copy_memory(dst, rbuf, rlen);
                }
            }
            raw::set_len(self, llen + rlen);
        }
    }

    /// Appends a character to the back of a string
    #[inline]
    fn push_char(&mut self, c: char) {
        assert!((c as uint) < MAX_UNICODE); // FIXME: #7609: should be enforced on all `char`
        let cur_len = self.len();
        self.reserve_at_least(cur_len + 4); // may use up to 4 bytes

        // Attempt to not use an intermediate buffer by just pushing bytes
        // directly onto this string.
        unsafe {
            let v = self.repr();
            let len = c.encode_utf8(cast::transmute(Slice {
                data: ((&(*v).data) as *u8).offset(cur_len as int),
                len: 4,
            }));
            raw::set_len(self, cur_len + len);
        }
    }

    /// Remove the final character from a string and return it
    ///
    /// # Failure
    ///
    /// If the string does not contain any characters
    fn pop_char(&mut self) -> char {
        let end = self.len();
        assert!(end > 0u);
        let CharRange {ch, next} = self.char_range_at_reverse(end);
        unsafe { raw::set_len(self, next); }
        return ch;
    }

    /// Remove the first character from a string and return it
    ///
    /// # Failure
    ///
    /// If the string does not contain any characters
    fn shift_char(&mut self) -> char {
        let CharRange {ch, next} = self.char_range_at(0u);
        *self = self.slice(next, self.len()).to_owned();
        return ch;
    }

    /// Prepend a char to a string
    fn unshift_char(&mut self, ch: char) {
        // This could be more efficient.
        let mut new_str = ~"";
        new_str.push_char(ch);
        new_str.push_str(*self);
        *self = new_str;
    }

    /// Concatenate two strings together.
    #[inline]
    fn append(self, rhs: &str) -> ~str {
        let mut new_str = self;
        new_str.push_str_no_overallocate(rhs);
        new_str
    }

    /// Reserves capacity for exactly `n` bytes in the given string, not including
    /// the null terminator.
    ///
    /// Assuming single-byte characters, the resulting string will be large
    /// enough to hold a string of length `n`. To account for the null terminator,
    /// the underlying buffer will have the size `n` + 1.
    ///
    /// If the capacity for `s` is already equal to or greater than the requested
    /// capacity, then no action is taken.
    ///
    /// # Arguments
    ///
    /// * s - A string
    /// * n - The number of bytes to reserve space for
    #[inline]
    fn reserve(&mut self, n: uint) {
        unsafe {
            let v: &mut ~[u8] = cast::transmute(self);
            (*v).reserve(n);
        }
    }

    /// Reserves capacity for at least `n` bytes in the given string.
    ///
    /// Assuming single-byte characters, the resulting string will be large
    /// enough to hold a string of length `n`. To account for the null terminator,
    /// the underlying buffer will have the size `n` + 1.
    ///
    /// This function will over-allocate in order to amortize the allocation costs
    /// in scenarios where the caller may need to repeatedly reserve additional
    /// space.
    ///
    /// If the capacity for `s` is already equal to or greater than the requested
    /// capacity, then no action is taken.
    ///
    /// # Arguments
    ///
    /// * s - A string
    /// * n - The number of bytes to reserve space for
    #[inline]
    fn reserve_at_least(&mut self, n: uint) {
        self.reserve(uint::next_power_of_two(n))
    }

    /// Returns the number of single-byte characters the string can hold without
    /// reallocating
    fn capacity(&self) -> uint {
        unsafe {
            let buf: &~[u8] = cast::transmute(self);
            buf.capacity()
        }
    }

    #[inline]
    fn as_mut_buf<T>(&mut self, f: &fn(*mut u8, uint) -> T) -> T {
        let v: &mut ~[u8] = unsafe { cast::transmute(self) };
        v.as_mut_buf(f)
    }
}

impl Clone for ~str {
    #[inline]
    fn clone(&self) -> ~str {
        self.to_owned()
    }
}

impl DeepClone for ~str {
    #[inline]
    fn deep_clone(&self) -> ~str {
        self.to_owned()
    }
}

impl Clone for @str {
    #[inline]
    fn clone(&self) -> @str {
        *self
    }
}

impl DeepClone for @str {
    #[inline]
    fn deep_clone(&self) -> @str {
        *self
    }
}

impl FromIterator<char> for ~str {
    #[inline]
    fn from_iterator<T: Iterator<char>>(iterator: &mut T) -> ~str {
        let (lower, _) = iterator.size_hint();
        let mut buf = with_capacity(lower);
        buf.extend(iterator);
        buf
    }
}

impl Extendable<char> for ~str {
    #[inline]
    fn extend<T: Iterator<char>>(&mut self, iterator: &mut T) {
        let (lower, _) = iterator.size_hint();
        let reserve = lower + self.len();
        self.reserve_at_least(reserve);
        for ch in *iterator {
            self.push_char(ch)
        }
    }
}

// This works because every lifetime is a sub-lifetime of 'static
impl<'self> Zero for &'self str {
    fn zero() -> &'self str { "" }
    fn is_zero(&self) -> bool { self.is_empty() }
}

impl Zero for ~str {
    fn zero() -> ~str { ~"" }
    fn is_zero(&self) -> bool { self.len() == 0 }
}

impl Zero for @str {
    fn zero() -> @str { @"" }
    fn is_zero(&self) -> bool { self.len() == 0 }
}

#[cfg(test)]
mod tests {
    use container::Container;
    use option::Some;
    use libc::c_char;
    use libc;
    use ptr;
    use str::*;
    use vec;
    use vec::{ImmutableVector, CopyableVector};
    use cmp::{TotalOrd, Less, Equal, Greater};

    #[test]
    fn test_eq() {
        assert!((eq(&~"", &~"")));
        assert!((eq(&~"foo", &~"foo")));
        assert!((!eq(&~"foo", &~"bar")));
    }

    #[test]
    fn test_eq_slice() {
        assert!((eq_slice("foobar".slice(0, 3), "foo")));
        assert!((eq_slice("barfoo".slice(3, 6), "foo")));
        assert!((!eq_slice("foo1", "foo2")));
    }

    #[test]
    fn test_le() {
        assert!("" <= "");
        assert!("" <= "foo");
        assert!("foo" <= "foo");
        assert!("foo" != "bar");
    }

    #[test]
    fn test_len() {
        assert_eq!("".len(), 0u);
        assert_eq!("hello world".len(), 11u);
        assert_eq!("\x63".len(), 1u);
        assert_eq!("\xa2".len(), 2u);
        assert_eq!("\u03c0".len(), 2u);
        assert_eq!("\u2620".len(), 3u);
        assert_eq!("\U0001d11e".len(), 4u);

        assert_eq!("".char_len(), 0u);
        assert_eq!("hello world".char_len(), 11u);
        assert_eq!("\x63".char_len(), 1u);
        assert_eq!("\xa2".char_len(), 1u);
        assert_eq!("\u03c0".char_len(), 1u);
        assert_eq!("\u2620".char_len(), 1u);
        assert_eq!("\U0001d11e".char_len(), 1u);
        assert_eq!("ประเทศไทย中华Việt Nam".char_len(), 19u);
    }

    #[test]
    fn test_find() {
        assert_eq!("hello".find('l'), Some(2u));
        assert_eq!("hello".find(|c:char| c == 'o'), Some(4u));
        assert!("hello".find('x').is_none());
        assert!("hello".find(|c:char| c == 'x').is_none());
        assert_eq!("ประเทศไทย中华Việt Nam".find('华'), Some(30u));
        assert_eq!("ประเทศไทย中华Việt Nam".find(|c: char| c == '华'), Some(30u));
    }

    #[test]
    fn test_rfind() {
        assert_eq!("hello".rfind('l'), Some(3u));
        assert_eq!("hello".rfind(|c:char| c == 'o'), Some(4u));
        assert!("hello".rfind('x').is_none());
        assert!("hello".rfind(|c:char| c == 'x').is_none());
        assert_eq!("ประเทศไทย中华Việt Nam".rfind('华'), Some(30u));
        assert_eq!("ประเทศไทย中华Việt Nam".rfind(|c: char| c == '华'), Some(30u));
    }

    #[test]
    fn test_push_str() {
        let mut s = ~"";
        s.push_str("");
        assert_eq!(s.slice_from(0), "");
        s.push_str("abc");
        assert_eq!(s.slice_from(0), "abc");
        s.push_str("ประเทศไทย中华Việt Nam");
        assert_eq!(s.slice_from(0), "abcประเทศไทย中华Việt Nam");
    }

    #[test]
    fn test_append() {
        let mut s = ~"";
        s = s.append("");
        assert_eq!(s.slice_from(0), "");
        s = s.append("abc");
        assert_eq!(s.slice_from(0), "abc");
        s = s.append("ประเทศไทย中华Việt Nam");
        assert_eq!(s.slice_from(0), "abcประเทศไทย中华Việt Nam");
    }

    #[test]
    fn test_pop_char() {
        let mut data = ~"ประเทศไทย中华";
        let cc = data.pop_char();
        assert_eq!(~"ประเทศไทย中", data);
        assert_eq!('华', cc);
    }

    #[test]
    fn test_pop_char_2() {
        let mut data2 = ~"华";
        let cc2 = data2.pop_char();
        assert_eq!(~"", data2);
        assert_eq!('华', cc2);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_pop_char_fail() {
        let mut data = ~"";
        let _cc3 = data.pop_char();
    }

    #[test]
    fn test_push_char() {
        let mut data = ~"ประเทศไทย中";
        data.push_char('华');
        data.push_char('b'); // 1 byte
        data.push_char('¢'); // 2 byte
        data.push_char('€'); // 3 byte
        data.push_char('𤭢'); // 4 byte
        assert_eq!(~"ประเทศไทย中华b¢€𤭢", data);
    }

    #[test]
    fn test_shift_char() {
        let mut data = ~"ประเทศไทย中";
        let cc = data.shift_char();
        assert_eq!(~"ระเทศไทย中", data);
        assert_eq!('ป', cc);
    }

    #[test]
    fn test_unshift_char() {
        let mut data = ~"ประเทศไทย中";
        data.unshift_char('华');
        assert_eq!(~"华ประเทศไทย中", data);
    }

    #[test]
    fn test_collect() {
        let empty = "";
        let s: ~str = empty.iter().collect();
        assert_eq!(empty, s.as_slice());
        let data = "ประเทศไทย中";
        let s: ~str = data.iter().collect();
        assert_eq!(data, s.as_slice());
    }

    #[test]
    fn test_extend() {
        let data = ~"ประเทศไทย中";
        let mut cpy = data.clone();
        let other = "abc";
        let mut it = other.iter();
        cpy.extend(&mut it);
        assert_eq!(cpy, data + other);
    }

    #[test]
    fn test_clear() {
        let mut empty = ~"";
        empty.clear();
        assert_eq!("", empty.as_slice());
        let mut data = ~"ประเทศไทย中";
        data.clear();
        assert_eq!("", data.as_slice());
        data.push_char('华');
        assert_eq!("华", data.as_slice());
    }

    #[test]
    fn test_find_str() {
        // byte positions
        assert_eq!("".find_str(""), Some(0u));
        assert!("banana".find_str("apple pie").is_none());

        let data = "abcabc";
        assert_eq!(data.slice(0u, 6u).find_str("ab"), Some(0u));
        assert_eq!(data.slice(2u, 6u).find_str("ab"), Some(3u - 2u));
        assert!(data.slice(2u, 4u).find_str("ab").is_none());

        let mut data = ~"ประเทศไทย中华Việt Nam";
        data = data + data;
        assert!(data.find_str("ไท华").is_none());
        assert_eq!(data.slice(0u, 43u).find_str(""), Some(0u));
        assert_eq!(data.slice(6u, 43u).find_str(""), Some(6u - 6u));

        assert_eq!(data.slice(0u, 43u).find_str("ประ"), Some( 0u));
        assert_eq!(data.slice(0u, 43u).find_str("ทศไ"), Some(12u));
        assert_eq!(data.slice(0u, 43u).find_str("ย中"), Some(24u));
        assert_eq!(data.slice(0u, 43u).find_str("iệt"), Some(34u));
        assert_eq!(data.slice(0u, 43u).find_str("Nam"), Some(40u));

        assert_eq!(data.slice(43u, 86u).find_str("ประ"), Some(43u - 43u));
        assert_eq!(data.slice(43u, 86u).find_str("ทศไ"), Some(55u - 43u));
        assert_eq!(data.slice(43u, 86u).find_str("ย中"), Some(67u - 43u));
        assert_eq!(data.slice(43u, 86u).find_str("iệt"), Some(77u - 43u));
        assert_eq!(data.slice(43u, 86u).find_str("Nam"), Some(83u - 43u));
    }

    #[test]
    fn test_slice_chars() {
        fn t(a: &str, b: &str, start: uint) {
            assert_eq!(a.slice_chars(start, start + b.char_len()), b);
        }
        t("hello", "llo", 2);
        t("hello", "el", 1);
        assert_eq!("ะเทศไท", "ประเทศไทย中华Việt Nam".slice_chars(2, 8));
    }

    #[test]
    fn test_concat() {
        fn t(v: &[~str], s: &str) {
            assert_eq!(v.concat(), s.to_str());
        }
        t([~"you", ~"know", ~"I'm", ~"no", ~"good"], "youknowI'mnogood");
        let v: &[~str] = [];
        t(v, "");
        t([~"hi"], "hi");
    }

    #[test]
    fn test_connect() {
        fn t(v: &[~str], sep: &str, s: &str) {
            assert_eq!(v.connect(sep), s.to_str());
        }
        t([~"you", ~"know", ~"I'm", ~"no", ~"good"],
          " ", "you know I'm no good");
        let v: &[~str] = [];
        t(v, " ", "");
        t([~"hi"], " ", "hi");
    }

    #[test]
    fn test_concat_slices() {
        fn t(v: &[&str], s: &str) {
            assert_eq!(v.concat(), s.to_str());
        }
        t(["you", "know", "I'm", "no", "good"], "youknowI'mnogood");
        let v: &[&str] = [];
        t(v, "");
        t(["hi"], "hi");
    }

    #[test]
    fn test_connect_slices() {
        fn t(v: &[&str], sep: &str, s: &str) {
            assert_eq!(v.connect(sep), s.to_str());
        }
        t(["you", "know", "I'm", "no", "good"],
          " ", "you know I'm no good");
        t([], " ", "");
        t(["hi"], " ", "hi");
    }

    #[test]
    fn test_repeat() {
        assert_eq!("x".repeat(4), ~"xxxx");
        assert_eq!("hi".repeat(4), ~"hihihihi");
        assert_eq!("ไท华".repeat(3), ~"ไท华ไท华ไท华");
        assert_eq!("".repeat(4), ~"");
        assert_eq!("hi".repeat(0), ~"");
    }

    #[test]
    fn test_unsafe_slice() {
        assert_eq!("ab", unsafe {raw::slice_bytes("abc", 0, 2)});
        assert_eq!("bc", unsafe {raw::slice_bytes("abc", 1, 3)});
        assert_eq!("", unsafe {raw::slice_bytes("abc", 1, 1)});
        fn a_million_letter_a() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 { rs.push_str("aaaaaaaaaa"); i += 1; }
            rs
        }
        fn half_a_million_letter_a() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 { rs.push_str("aaaaa"); i += 1; }
            rs
        }
        let letters = a_million_letter_a();
        assert!(half_a_million_letter_a() ==
            unsafe {raw::slice_bytes(letters, 0u, 500000)}.to_owned());
    }

    #[test]
    fn test_starts_with() {
        assert!(("".starts_with("")));
        assert!(("abc".starts_with("")));
        assert!(("abc".starts_with("a")));
        assert!((!"a".starts_with("abc")));
        assert!((!"".starts_with("abc")));
    }

    #[test]
    fn test_ends_with() {
        assert!(("".ends_with("")));
        assert!(("abc".ends_with("")));
        assert!(("abc".ends_with("c")));
        assert!((!"a".ends_with("abc")));
        assert!((!"".ends_with("abc")));
    }

    #[test]
    fn test_is_empty() {
        assert!("".is_empty());
        assert!(!"a".is_empty());
    }

    #[test]
    fn test_replace() {
        let a = "a";
        assert_eq!("".replace(a, "b"), ~"");
        assert_eq!("a".replace(a, "b"), ~"b");
        assert_eq!("ab".replace(a, "b"), ~"bb");
        let test = "test";
        assert!(" test test ".replace(test, "toast") ==
            ~" toast toast ");
        assert_eq!(" test test ".replace(test, ""), ~"   ");
    }

    #[test]
    fn test_replace_2a() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let a = ~"ประเ";
        let A = ~"دولة الكويتทศไทย中华";
        assert_eq!(data.replace(a, repl), A);
    }

    #[test]
    fn test_replace_2b() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let b = ~"ะเ";
        let B = ~"ปรدولة الكويتทศไทย中华";
        assert_eq!(data.replace(b,   repl), B);
    }

    #[test]
    fn test_replace_2c() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let c = ~"中华";
        let C = ~"ประเทศไทยدولة الكويت";
        assert_eq!(data.replace(c, repl), C);
    }

    #[test]
    fn test_replace_2d() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let d = ~"ไท华";
        assert_eq!(data.replace(d, repl), data);
    }

    #[test]
    fn test_slice() {
        assert_eq!("ab", "abc".slice(0, 2));
        assert_eq!("bc", "abc".slice(1, 3));
        assert_eq!("", "abc".slice(1, 1));
        assert_eq!("\u65e5", "\u65e5\u672c".slice(0, 3));

        let data = "ประเทศไทย中华";
        assert_eq!("ป", data.slice(0, 3));
        assert_eq!("ร", data.slice(3, 6));
        assert_eq!("", data.slice(3, 3));
        assert_eq!("华", data.slice(30, 33));

        fn a_million_letter_X() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 {
                push_str(&mut rs, "华华华华华华华华华华");
                i += 1;
            }
            rs
        }
        fn half_a_million_letter_X() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 { push_str(&mut rs, "华华华华华"); i += 1; }
            rs
        }
        let letters = a_million_letter_X();
        assert!(half_a_million_letter_X() ==
            letters.slice(0u, 3u * 500000u).to_owned());
    }

    #[test]
    fn test_slice_2() {
        let ss = "中华Việt Nam";

        assert_eq!("华", ss.slice(3u, 6u));
        assert_eq!("Việt Nam", ss.slice(6u, 16u));

        assert_eq!("ab", "abc".slice(0u, 2u));
        assert_eq!("bc", "abc".slice(1u, 3u));
        assert_eq!("", "abc".slice(1u, 1u));

        assert_eq!("中", ss.slice(0u, 3u));
        assert_eq!("华V", ss.slice(3u, 7u));
        assert_eq!("", ss.slice(3u, 3u));
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
    #[ignore(cfg(windows))]
    fn test_slice_fail() {
        "中华Việt Nam".slice(0u, 2u);
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
    fn test_trim_left_chars() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_left_chars(&v), " *** foo *** ");
        assert_eq!(" *** foo *** ".trim_left_chars(& &['*', ' ']), "foo *** ");
        assert_eq!(" ***  *** ".trim_left_chars(& &['*', ' ']), "");
        assert_eq!("foo *** ".trim_left_chars(& &['*', ' ']), "foo *** ");

        assert_eq!("11foo1bar11".trim_left_chars(&'1'), "foo1bar11");
        assert_eq!("12foo1bar12".trim_left_chars(& &['1', '2']), "foo1bar12");
        assert_eq!("123foo1bar123".trim_left_chars(&|c: char| c.is_digit()), "foo1bar123");
    }

    #[test]
    fn test_trim_right_chars() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_right_chars(&v), " *** foo *** ");
        assert_eq!(" *** foo *** ".trim_right_chars(& &['*', ' ']), " *** foo");
        assert_eq!(" ***  *** ".trim_right_chars(& &['*', ' ']), "");
        assert_eq!(" *** foo".trim_right_chars(& &['*', ' ']), " *** foo");

        assert_eq!("11foo1bar11".trim_right_chars(&'1'), "11foo1bar");
        assert_eq!("12foo1bar12".trim_right_chars(& &['1', '2']), "12foo1bar");
        assert_eq!("123foo1bar123".trim_right_chars(&|c: char| c.is_digit()), "123foo1bar");
    }

    #[test]
    fn test_trim_chars() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_chars(&v), " *** foo *** ");
        assert_eq!(" *** foo *** ".trim_chars(& &['*', ' ']), "foo");
        assert_eq!(" ***  *** ".trim_chars(& &['*', ' ']), "");
        assert_eq!("foo".trim_chars(& &['*', ' ']), "foo");

        assert_eq!("11foo1bar11".trim_chars(&'1'), "foo1bar");
        assert_eq!("12foo1bar12".trim_chars(& &['1', '2']), "foo1bar");
        assert_eq!("123foo1bar123".trim_chars(&|c: char| c.is_digit()), "foo1bar");
    }

    #[test]
    fn test_trim_left() {
        assert_eq!("".trim_left(), "");
        assert_eq!("a".trim_left(), "a");
        assert_eq!("    ".trim_left(), "");
        assert_eq!("     blah".trim_left(), "blah");
        assert_eq!("   \u3000  wut".trim_left(), "wut");
        assert_eq!("hey ".trim_left(), "hey ");
    }

    #[test]
    fn test_trim_right() {
        assert_eq!("".trim_right(), "");
        assert_eq!("a".trim_right(), "a");
        assert_eq!("    ".trim_right(), "");
        assert_eq!("blah     ".trim_right(), "blah");
        assert_eq!("wut   \u3000  ".trim_right(), "wut");
        assert_eq!(" hey".trim_right(), " hey");
    }

    #[test]
    fn test_trim() {
        assert_eq!("".trim(), "");
        assert_eq!("a".trim(), "a");
        assert_eq!("    ".trim(), "");
        assert_eq!("    blah     ".trim(), "blah");
        assert_eq!("\nwut   \u3000  ".trim(), "wut");
        assert_eq!(" hey dude ".trim(), "hey dude");
    }

    #[test]
    fn test_is_whitespace() {
        assert!("".is_whitespace());
        assert!(" ".is_whitespace());
        assert!("\u2009".is_whitespace()); // Thin space
        assert!("  \n\t   ".is_whitespace());
        assert!(!"   _   ".is_whitespace());
    }

    #[test]
    fn test_push_byte() {
        let mut s = ~"ABC";
        unsafe{raw::push_byte(&mut s, 'D' as u8)};
        assert_eq!(s, ~"ABCD");
    }

    #[test]
    fn test_shift_byte() {
        let mut s = ~"ABC";
        let b = unsafe{raw::shift_byte(&mut s)};
        assert_eq!(s, ~"BC");
        assert_eq!(b, 65u8);
    }

    #[test]
    fn test_pop_byte() {
        let mut s = ~"ABC";
        let b = unsafe{raw::pop_byte(&mut s)};
        assert_eq!(s, ~"AB");
        assert_eq!(b, 67u8);
    }

    #[test]
    fn test_unsafe_from_bytes() {
        let a = ~[65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8];
        let b = unsafe { raw::from_bytes(a) };
        assert_eq!(b, ~"AAAAAAA");
    }

    #[test]
    fn test_from_bytes() {
        let ss = ~"ศไทย中华Việt Nam";
        let bb = ~[0xe0_u8, 0xb8_u8, 0xa8_u8,
                  0xe0_u8, 0xb9_u8, 0x84_u8,
                  0xe0_u8, 0xb8_u8, 0x97_u8,
                  0xe0_u8, 0xb8_u8, 0xa2_u8,
                  0xe4_u8, 0xb8_u8, 0xad_u8,
                  0xe5_u8, 0x8d_u8, 0x8e_u8,
                  0x56_u8, 0x69_u8, 0xe1_u8,
                  0xbb_u8, 0x87_u8, 0x74_u8,
                  0x20_u8, 0x4e_u8, 0x61_u8,
                  0x6d_u8];


        assert_eq!(ss, from_bytes(bb));
        assert_eq!(~"𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰",
                   from_bytes(bytes!("𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰")));
    }

    #[test]
    fn test_is_utf8() {
        assert!(!is_utf8([0xc0, 0x80]));
        assert!(!is_utf8([0xc0, 0xae]));
        assert!(!is_utf8([0xe0, 0x80, 0x80]));
        assert!(!is_utf8([0xe0, 0x80, 0xaf]));
        assert!(!is_utf8([0xe0, 0x81, 0x81]));
        assert!(!is_utf8([0xf0, 0x82, 0x82, 0xac]));
        assert!(!is_utf8([0xf4, 0x90, 0x80, 0x80]));

        assert!(is_utf8([0xC2, 0x80]));
        assert!(is_utf8([0xDF, 0xBF]));
        assert!(is_utf8([0xE0, 0xA0, 0x80]));
        assert!(is_utf8([0xEF, 0xBF, 0xBF]));
        assert!(is_utf8([0xF0, 0x90, 0x80, 0x80]));
        assert!(is_utf8([0xF4, 0x8F, 0xBF, 0xBF]));
    }


    #[test]
    #[ignore(cfg(windows))]
    fn test_from_bytes_fail() {
        use str::not_utf8::cond;

        let bb = ~[0xff_u8, 0xb8_u8, 0xa8_u8,
                  0xe0_u8, 0xb9_u8, 0x84_u8,
                  0xe0_u8, 0xb8_u8, 0x97_u8,
                  0xe0_u8, 0xb8_u8, 0xa2_u8,
                  0xe4_u8, 0xb8_u8, 0xad_u8,
                  0xe5_u8, 0x8d_u8, 0x8e_u8,
                  0x56_u8, 0x69_u8, 0xe1_u8,
                  0xbb_u8, 0x87_u8, 0x74_u8,
                  0x20_u8, 0x4e_u8, 0x61_u8,
                  0x6d_u8];

        let mut error_happened = false;
        let _x = do cond.trap(|err| {
            assert_eq!(err, ~"from_bytes: input is not UTF-8; first bad byte is 255");
            error_happened = true;
            ~""
        }).inside {
            from_bytes(bb)
        };
        assert!(error_happened);
    }

    #[test]
    fn test_raw_from_c_str() {
        unsafe {
            let a = ~[65, 65, 65, 65, 65, 65, 65, 0];
            let b = vec::raw::to_ptr(a);
            let c = raw::from_c_str(b);
            assert_eq!(c, ~"AAAAAAA");
        }
    }

    #[test]
    fn test_as_bytes() {
        // no null
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        assert_eq!("".as_bytes(), &[]);
        assert_eq!("abc".as_bytes(), &['a' as u8, 'b' as u8, 'c' as u8]);
        assert_eq!("ศไทย中华Việt Nam".as_bytes(), v);
    }

    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_as_bytes_fail() {
        // Don't double free. (I'm not sure if this exercises the
        // original problem code path anymore.)
        let s = ~"";
        let _bytes = s.as_bytes();
        fail!();
    }

    #[test]
    fn test_as_imm_buf() {
        do "".as_imm_buf |_, len| {
            assert_eq!(len, 0);
        }

        do "hello".as_imm_buf |buf, len| {
            assert_eq!(len, 5);
            unsafe {
                assert_eq!(*ptr::offset(buf, 0), 'h' as u8);
                assert_eq!(*ptr::offset(buf, 1), 'e' as u8);
                assert_eq!(*ptr::offset(buf, 2), 'l' as u8);
                assert_eq!(*ptr::offset(buf, 3), 'l' as u8);
                assert_eq!(*ptr::offset(buf, 4), 'o' as u8);
            }
        }
    }

    #[test]
    fn test_subslice_offset() {
        let a = "kernelsprite";
        let b = a.slice(7, a.len());
        let c = a.slice(0, a.len() - 6);
        assert_eq!(a.subslice_offset(b), 7);
        assert_eq!(a.subslice_offset(c), 0);

        let string = "a\nb\nc";
        let mut lines = ~[];
        for line in string.line_iter() { lines.push(line) }
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
        let s1: ~str = ~"All mimsy were the borogoves";

        let v: ~[u8] = s1.as_bytes().to_owned();
        let s2: ~str = from_bytes(v);
        let mut i: uint = 0u;
        let n1: uint = s1.len();
        let n2: uint = v.len();
        assert_eq!(n1, n2);
        while i < n1 {
            let a: u8 = s1[i];
            let b: u8 = s2[i];
            debug!(a);
            debug!(b);
            assert_eq!(a, b);
            i += 1u;
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

        let data = ~"ประเทศไทย中华Việt Nam";
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
    fn test_map() {
        assert_eq!(~"", "".map_chars(|c| unsafe {libc::toupper(c as c_char)} as char));
        assert_eq!(~"YMCA", "ymca".map_chars(|c| unsafe {libc::toupper(c as c_char)} as char));
    }

    #[test]
    fn test_utf16() {
        let pairs =
            [(~"𐍅𐌿𐌻𐍆𐌹𐌻𐌰\n",
              ~[0xd800_u16, 0xdf45_u16, 0xd800_u16, 0xdf3f_u16,
                0xd800_u16, 0xdf3b_u16, 0xd800_u16, 0xdf46_u16,
                0xd800_u16, 0xdf39_u16, 0xd800_u16, 0xdf3b_u16,
                0xd800_u16, 0xdf30_u16, 0x000a_u16]),

             (~"𐐒𐑉𐐮𐑀𐐲𐑋 𐐏𐐲𐑍\n",
              ~[0xd801_u16, 0xdc12_u16, 0xd801_u16,
                0xdc49_u16, 0xd801_u16, 0xdc2e_u16, 0xd801_u16,
                0xdc40_u16, 0xd801_u16, 0xdc32_u16, 0xd801_u16,
                0xdc4b_u16, 0x0020_u16, 0xd801_u16, 0xdc0f_u16,
                0xd801_u16, 0xdc32_u16, 0xd801_u16, 0xdc4d_u16,
                0x000a_u16]),

             (~"𐌀𐌖𐌋𐌄𐌑𐌉·𐌌𐌄𐌕𐌄𐌋𐌉𐌑\n",
              ~[0xd800_u16, 0xdf00_u16, 0xd800_u16, 0xdf16_u16,
                0xd800_u16, 0xdf0b_u16, 0xd800_u16, 0xdf04_u16,
                0xd800_u16, 0xdf11_u16, 0xd800_u16, 0xdf09_u16,
                0x00b7_u16, 0xd800_u16, 0xdf0c_u16, 0xd800_u16,
                0xdf04_u16, 0xd800_u16, 0xdf15_u16, 0xd800_u16,
                0xdf04_u16, 0xd800_u16, 0xdf0b_u16, 0xd800_u16,
                0xdf09_u16, 0xd800_u16, 0xdf11_u16, 0x000a_u16 ]),

             (~"𐒋𐒘𐒈𐒑𐒛𐒒 𐒕𐒓 𐒈𐒚𐒍 𐒏𐒜𐒒𐒖𐒆 𐒕𐒆\n",
              ~[0xd801_u16, 0xdc8b_u16, 0xd801_u16, 0xdc98_u16,
                0xd801_u16, 0xdc88_u16, 0xd801_u16, 0xdc91_u16,
                0xd801_u16, 0xdc9b_u16, 0xd801_u16, 0xdc92_u16,
                0x0020_u16, 0xd801_u16, 0xdc95_u16, 0xd801_u16,
                0xdc93_u16, 0x0020_u16, 0xd801_u16, 0xdc88_u16,
                0xd801_u16, 0xdc9a_u16, 0xd801_u16, 0xdc8d_u16,
                0x0020_u16, 0xd801_u16, 0xdc8f_u16, 0xd801_u16,
                0xdc9c_u16, 0xd801_u16, 0xdc92_u16, 0xd801_u16,
                0xdc96_u16, 0xd801_u16, 0xdc86_u16, 0x0020_u16,
                0xd801_u16, 0xdc95_u16, 0xd801_u16, 0xdc86_u16,
                0x000a_u16 ]) ];

        for p in pairs.iter() {
            let (s, u) = (*p).clone();
            assert!(s.to_utf16() == u);
            assert!(from_utf16(u) == s);
            assert!(from_utf16(s.to_utf16()) == s);
            assert!(from_utf16(u).to_utf16() == u);
        }
    }

    #[test]
    fn test_char_at() {
        let s = ~"ศไทย中华Việt Nam";
        let v = ~['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
        let mut pos = 0;
        for ch in v.iter() {
            assert!(s.char_at(pos) == *ch);
            pos += from_char(*ch).len();
        }
    }

    #[test]
    fn test_char_at_reverse() {
        let s = ~"ศไทย中华Việt Nam";
        let v = ~['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
        let mut pos = s.len();
        for ch in v.rev_iter() {
            assert!(s.char_at_reverse(pos) == *ch);
            pos -= from_char(*ch).len();
        }
    }

    #[test]
    fn test_escape_unicode() {
        assert_eq!("abc".escape_unicode(), ~"\\x61\\x62\\x63");
        assert_eq!("a c".escape_unicode(), ~"\\x61\\x20\\x63");
        assert_eq!("\r\n\t".escape_unicode(), ~"\\x0d\\x0a\\x09");
        assert_eq!("'\"\\".escape_unicode(), ~"\\x27\\x22\\x5c");
        assert_eq!("\x00\x01\xfe\xff".escape_unicode(), ~"\\x00\\x01\\xfe\\xff");
        assert_eq!("\u0100\uffff".escape_unicode(), ~"\\u0100\\uffff");
        assert_eq!("\U00010000\U0010ffff".escape_unicode(), ~"\\U00010000\\U0010ffff");
        assert_eq!("ab\ufb00".escape_unicode(), ~"\\x61\\x62\\ufb00");
        assert_eq!("\U0001d4ea\r".escape_unicode(), ~"\\U0001d4ea\\x0d");
    }

    #[test]
    fn test_escape_default() {
        assert_eq!("abc".escape_default(), ~"abc");
        assert_eq!("a c".escape_default(), ~"a c");
        assert_eq!("\r\n\t".escape_default(), ~"\\r\\n\\t");
        assert_eq!("'\"\\".escape_default(), ~"\\'\\\"\\\\");
        assert_eq!("\u0100\uffff".escape_default(), ~"\\u0100\\uffff");
        assert_eq!("\U00010000\U0010ffff".escape_default(), ~"\\U00010000\\U0010ffff");
        assert_eq!("ab\ufb00".escape_default(), ~"ab\\ufb00");
        assert_eq!("\U0001d4ea\r".escape_default(), ~"\\U0001d4ea\\r");
    }

    #[test]
    fn test_to_managed() {
        assert_eq!("abc".to_managed(), @"abc");
        assert_eq!("abcdef".slice(1, 5).to_managed(), @"bcde");
    }

    #[test]
    fn test_total_ord() {
        "1234".cmp(& &"123") == Greater;
        "123".cmp(& &"1234") == Less;
        "1234".cmp(& &"1234") == Equal;
        "12345555".cmp(& &"123456") == Less;
        "22".cmp(& &"1234") == Greater;
    }

    #[test]
    fn test_char_range_at() {
        let data = ~"b¢€𤭢𤭢€¢b";
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
    fn test_add() {
        #[allow(unnecessary_allocation)];
        macro_rules! t (
            ($s1:expr, $s2:expr, $e:expr) => { {
                let s1 = $s1;
                let s2 = $s2;
                let e = $e;
                assert_eq!(s1 + s2, e.to_owned());
                assert_eq!(s1.to_owned() + s2, e.to_owned());
                assert_eq!(s1.to_managed() + s2, e.to_owned());
            } }
        );

        t!("foo",  "bar", "foobar");
        t!("foo", @"bar", "foobar");
        t!("foo", ~"bar", "foobar");
        t!("ศไทย中",  "华Việt Nam", "ศไทย中华Việt Nam");
        t!("ศไทย中", @"华Việt Nam", "ศไทย中华Việt Nam");
        t!("ศไทย中", ~"华Việt Nam", "ศไทย中华Việt Nam");
    }

    #[test]
    fn test_iterator() {
        use iterator::*;
        let s = ~"ศไทย中华Việt Nam";
        let v = ~['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

        let mut pos = 0;
        let mut it = s.iter();

        for c in it {
            assert_eq!(c, v[pos]);
            pos += 1;
        }
        assert_eq!(pos, v.len());
    }

    #[test]
    fn test_rev_iterator() {
        use iterator::*;
        let s = ~"ศไทย中华Việt Nam";
        let v = ~['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

        let mut pos = 0;
        let mut it = s.rev_iter();

        for c in it {
            assert_eq!(c, v[pos]);
            pos += 1;
        }
        assert_eq!(pos, v.len());
    }

    #[test]
    fn test_byte_iterator() {
        let s = ~"ศไทย中华Việt Nam";
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        let mut pos = 0;

        for b in s.byte_iter() {
            assert_eq!(b, v[pos]);
            pos += 1;
        }
    }

    #[test]
    fn test_byte_rev_iterator() {
        let s = ~"ศไทย中华Việt Nam";
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        let mut pos = v.len();

        for b in s.byte_rev_iter() {
            pos -= 1;
            assert_eq!(b, v[pos]);
        }
    }

    #[test]
    fn test_char_offset_iterator() {
        use iterator::*;
        let s = "ศไทย中华Việt Nam";
        let p = [0, 3, 6, 9, 12, 15, 18, 19, 20, 23, 24, 25, 26, 27];
        let v = ['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

        let mut pos = 0;
        let mut it = s.char_offset_iter();

        for c in it {
            assert_eq!(c, (p[pos], v[pos]));
            pos += 1;
        }
        assert_eq!(pos, v.len());
        assert_eq!(pos, p.len());
    }

    #[test]
    fn test_char_offset_rev_iterator() {
        use iterator::*;
        let s = "ศไทย中华Việt Nam";
        let p = [27, 26, 25, 24, 23, 20, 19, 18, 15, 12, 9, 6, 3, 0];
        let v = ['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

        let mut pos = 0;
        let mut it = s.char_offset_rev_iter();

        for c in it {
            assert_eq!(c, (p[pos], v[pos]));
            pos += 1;
        }
        assert_eq!(pos, v.len());
        assert_eq!(pos, p.len());
    }

    #[test]
    fn test_split_char_iterator() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: ~[&str] = data.split_iter(' ').collect();
        assert_eq!(split, ~["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let split: ~[&str] = data.split_iter(|c: char| c == ' ').collect();
        assert_eq!(split, ~["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        // Unicode
        let split: ~[&str] = data.split_iter('ä').collect();
        assert_eq!(split, ~["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let split: ~[&str] = data.split_iter(|c: char| c == 'ä').collect();
        assert_eq!(split, ~["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);
    }

    #[test]
    fn test_splitn_char_iterator() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: ~[&str] = data.splitn_iter(' ', 3).collect();
        assert_eq!(split, ~["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

        let split: ~[&str] = data.splitn_iter(|c: char| c == ' ', 3).collect();
        assert_eq!(split, ~["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

        // Unicode
        let split: ~[&str] = data.splitn_iter('ä', 3).collect();
        assert_eq!(split, ~["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);

        let split: ~[&str] = data.splitn_iter(|c: char| c == 'ä', 3).collect();
        assert_eq!(split, ~["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);
    }

    #[test]
    fn test_split_char_iterator_no_trailing() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: ~[&str] = data.split_options_iter('\n', 1000, true).collect();
        assert_eq!(split, ~["", "Märy häd ä little lämb", "Little lämb", ""]);

        let split: ~[&str] = data.split_options_iter('\n', 1000, false).collect();
        assert_eq!(split, ~["", "Märy häd ä little lämb", "Little lämb"]);
    }

    #[test]
    fn test_word_iter() {
        let data = "\n \tMäry   häd\tä  little lämb\nLittle lämb\n";
        let words: ~[&str] = data.word_iter().collect();
        assert_eq!(words, ~["Märy", "häd", "ä", "little", "lämb", "Little", "lämb"])
    }

    #[test]
    fn test_line_iter() {
        let data = "\nMäry häd ä little lämb\n\nLittle lämb\n";
        let lines: ~[&str] = data.line_iter().collect();
        assert_eq!(lines, ~["", "Märy häd ä little lämb", "", "Little lämb"]);

        let data = "\nMäry häd ä little lämb\n\nLittle lämb"; // no trailing \n
        let lines: ~[&str] = data.line_iter().collect();
        assert_eq!(lines, ~["", "Märy häd ä little lämb", "", "Little lämb"]);
    }

    #[test]
    fn test_split_str_iterator() {
        fn t<'a>(s: &str, sep: &'a str, u: ~[&str]) {
            let v: ~[&str] = s.split_str_iter(sep).collect();
            assert_eq!(v, u);
        }
        t("--1233345--", "12345", ~["--1233345--"]);
        t("abc::hello::there", "::", ~["abc", "hello", "there"]);
        t("::hello::there", "::", ~["", "hello", "there"]);
        t("hello::there::", "::", ~["hello", "there", ""]);
        t("::hello::there::", "::", ~["", "hello", "there", ""]);
        t("ประเทศไทย中华Việt Nam", "中华", ~["ประเทศไทย", "Việt Nam"]);
        t("zzXXXzzYYYzz", "zz", ~["", "XXX", "YYY", ""]);
        t("zzXXXzYYYz", "XXX", ~["zz", "zYYYz"]);
        t(".XXX.YYY.", ".", ~["", "XXX", "YYY", ""]);
        t("", ".", ~[""]);
        t("zz", "zz", ~["",""]);
        t("ok", "z", ~["ok"]);
        t("zzz", "zz", ~["","z"]);
        t("zzzzz", "zz", ~["","","z"]);
    }

    #[test]
    fn test_str_zero() {
        use num::Zero;
        fn t<S: Zero + Str>() {
            let s: S = Zero::zero();
            assert_eq!(s.as_slice(), "");
            assert!(s.is_zero());
        }

        t::<&str>();
        t::<@str>();
        t::<~str>();
    }

    #[test]
    fn test_str_container() {
        fn sum_len<S: Container>(v: &[S]) -> uint {
            v.iter().map(|x| x.len()).sum()
        }

        let s = ~"01234";
        assert_eq!(5, sum_len(["012", "", "34"]));
        assert_eq!(5, sum_len([@"01", @"2", @"34", @""]));
        assert_eq!(5, sum_len([~"01", ~"2", ~"34", ~""]));
        assert_eq!(5, sum_len([s.as_slice()]));
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use super::*;

    #[bench]
    fn is_utf8_100_ascii(bh: &mut BenchHarness) {

        let s = bytes!("Hello there, the quick brown fox jumped over the lazy dog! \
                        Lorem ipsum dolor sit amet, consectetur. ");

        assert_eq!(100, s.len());
        do bh.iter {
            is_utf8(s);
        }
    }

    #[bench]
    fn is_utf8_100_multibyte(bh: &mut BenchHarness) {
        let s = bytes!("𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰");
        assert_eq!(100, s.len());
        do bh.iter {
            is_utf8(s);
        }
    }

    #[bench]
    fn map_chars_100_ascii(bh: &mut BenchHarness) {
        let s = "HelloHelloHelloHelloHelloHelloHelloHelloHelloHello\
                 HelloHelloHelloHelloHelloHelloHelloHelloHelloHello";
        do bh.iter {
            s.map_chars(|c| ((c as uint) + 1) as char);
        }
    }

    #[bench]
    fn map_chars_100_multibytes(bh: &mut BenchHarness) {
        let s = "𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑\
                 𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑\
                 𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑\
                 𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑𐌀𐌖𐌋𐌄𐌑";
        do bh.iter {
            s.map_chars(|c| ((c as uint) + 1) as char);
        }
    }

    #[bench]
    fn bench_with_capacity(bh: &mut BenchHarness) {
        do bh.iter {
            with_capacity(100);
        }
    }
}
