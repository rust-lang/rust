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

/*!

Unicode string manipulation (`str` type)

# Basic Usage

Rust's string type is one of the core primitive types of the language. While
represented by the name `str`, the name `str` is not actually a valid type in
Rust. Each string must also be decorated with a pointer. `String` is used
for an owned string, so there is only one commonly-used `str` type in Rust:
`&str`.

`&str` is the borrowed string type. This type of string can only be created
from other strings, unless it is a static string (see below). As the word
"borrowed" implies, this type of string is owned elsewhere, and this string
cannot be moved out of.

As an example, here's some code that uses a string.

```rust
fn main() {
    let borrowed_string = "This string is borrowed with the 'static lifetime";
}
```

From the example above, you can see that Rust's string literals have the
`'static` lifetime. This is akin to C's concept of a static string.

String literals are allocated statically in the rodata of the
executable/library. The string then has the type `&'static str` meaning that
the string is valid for the `'static` lifetime, otherwise known as the
lifetime of the entire program. As can be inferred from the type, these static
strings are not mutable.

# Representation

Rust's string type, `str`, is a sequence of unicode scalar values encoded as a
stream of UTF-8 bytes. All strings are guaranteed to be validly encoded UTF-8
sequences. Additionally, strings are not null-terminated and can contain null
bytes.

The actual representation of strings have direct mappings to vectors: `&str`
is the same as `&[u8]`.

*/

#![doc(primitive = "str")]

use core::default::Default;
use core::fmt;
use core::cmp;
use core::iter::AdditiveIterator;
use core::mem;
use core::prelude::{Char, Clone, Collection, Eq, Equiv, ImmutableSlice};
use core::prelude::{Iterator, MutableSlice, None, Option, Ord, Ordering};
use core::prelude::{PartialEq, PartialOrd, Result, Slice, Some, Tuple2};
use core::prelude::{range};

use {Deque, MutableSeq};
use hash;
use ringbuf::RingBuf;
use string::String;
use unicode;
use vec::Vec;

pub use core::str::{from_utf8, CharEq, Chars, CharOffsets};
pub use core::str::{Bytes, CharSplits};
pub use core::str::{CharSplitsN, AnyLines, MatchIndices, StrSplits};
pub use core::str::{eq_slice, is_utf8, is_utf16, Utf16Items};
pub use core::str::{Utf16Item, ScalarValue, LoneSurrogate, utf16_items};
pub use core::str::{truncate_utf16_at_nul, utf8_char_width, CharRange};
pub use core::str::{Str, StrSlice};
pub use unicode::str::{UnicodeStrSlice, Words, Graphemes, GraphemeIndices};

/*
Section: Creating a string
*/

/// Deprecated. Replaced by `String::from_utf8`
#[deprecated = "Replaced by `String::from_utf8`"]
pub fn from_utf8_owned(vv: Vec<u8>) -> Result<String, Vec<u8>> {
    String::from_utf8(vv)
}

/// Deprecated. Replaced by `String::from_byte`
#[deprecated = "Replaced by String::from_byte"]
pub fn from_byte(b: u8) -> String {
    assert!(b < 128u8);
    String::from_char(1, b as char)
}

/// Deprecated. Use `String::from_char` or `char::to_string()` instead
#[deprecated = "use String::from_char or char.to_string()"]
pub fn from_char(ch: char) -> String {
    String::from_char(1, ch)
}

/// Deprecated. Replaced by `String::from_chars`
#[deprecated = "use String::from_chars instead"]
pub fn from_chars(chs: &[char]) -> String {
    chs.iter().map(|c| *c).collect()
}

/// Methods for vectors of strings
pub trait StrVector {
    /// Concatenate a vector of strings.
    ///
    /// # Example
    ///
    /// ```rust
    /// let first = "Restaurant at the End of the".to_string();
    /// let second = " Universe".to_string();
    /// let string_vec = vec![first, second];
    /// assert_eq!(string_vec.concat(), "Restaurant at the End of the Universe".to_string());
    /// ```
    fn concat(&self) -> String;

    /// Concatenate a vector of strings, placing a given separator between each.
    ///
    /// # Example
    ///
    /// ```rust
    /// let first = "Roast".to_string();
    /// let second = "Sirloin Steak".to_string();
    /// let string_vec = vec![first, second];
    /// assert_eq!(string_vec.connect(", "), "Roast, Sirloin Steak".to_string());
    /// ```
    fn connect(&self, sep: &str) -> String;
}

impl<'a, S: Str> StrVector for &'a [S] {
    fn concat(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        // `len` calculation may overflow but push_str will check boundaries
        let len = self.iter().map(|s| s.as_slice().len()).sum();

        let mut result = String::with_capacity(len);

        for s in self.iter() {
            result.push_str(s.as_slice())
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
            + self.iter().map(|s| s.as_slice().len()).sum();
        let mut result = String::with_capacity(len);
        let mut first = true;

        for s in self.iter() {
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

impl<'a, S: Str> StrVector for Vec<S> {
    #[inline]
    fn concat(&self) -> String {
        self.as_slice().concat()
    }

    #[inline]
    fn connect(&self, sep: &str) -> String {
        self.as_slice().connect(sep)
    }
}

/*
Section: Iterators
*/

// Helper functions used for Unicode normalization
fn canonical_sort(comb: &mut [(char, u8)]) {
    let len = comb.len();
    for i in range(0, len) {
        let mut swapped = false;
        for j in range(1, len-i) {
            let class_a = *comb[j-1].ref1();
            let class_b = *comb[j].ref1();
            if class_a != 0 && class_b != 0 && class_a > class_b {
                comb.swap(j-1, j);
                swapped = true;
            }
        }
        if !swapped { break; }
    }
}

#[deriving(Clone)]
enum DecompositionType {
    Canonical,
    Compatible
}

/// External iterator for a string's decomposition's characters.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct Decompositions<'a> {
    kind: DecompositionType,
    iter: Chars<'a>,
    buffer: Vec<(char, u8)>,
    sorted: bool
}

impl<'a> Iterator<char> for Decompositions<'a> {
    #[inline]
    fn next(&mut self) -> Option<char> {
        match self.buffer.as_slice().head() {
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

        let decomposer = match self.kind {
            Canonical => unicode::char::decompose_canonical,
            Compatible => unicode::char::decompose_compatible
        };

        if !self.sorted {
            for ch in self.iter {
                let buffer = &mut self.buffer;
                let sorted = &mut self.sorted;
                decomposer(ch, |d| {
                    let class = unicode::char::canonical_combining_class(d);
                    if class == 0 && !*sorted {
                        canonical_sort(buffer.as_mut_slice());
                        *sorted = true;
                    }
                    buffer.push((d, class));
                });
                if *sorted { break }
            }
        }

        if !self.sorted {
            canonical_sort(self.buffer.as_mut_slice());
            self.sorted = true;
        }

        match self.buffer.remove(0) {
            Some((c, 0)) => {
                self.sorted = false;
                Some(c)
            }
            Some((c, _)) => Some(c),
            None => None
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lower, _) = self.iter.size_hint();
        (lower, None)
    }
}

#[deriving(Clone)]
enum RecompositionState {
    Composing,
    Purging,
    Finished
}

/// External iterator for a string's recomposition's characters.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct Recompositions<'a> {
    iter: Decompositions<'a>,
    state: RecompositionState,
    buffer: RingBuf<char>,
    composee: Option<char>,
    last_ccc: Option<u8>
}

impl<'a> Iterator<char> for Recompositions<'a> {
    #[inline]
    fn next(&mut self) -> Option<char> {
        loop {
            match self.state {
                Composing => {
                    for ch in self.iter {
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
                                        self.buffer.push(ch);
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
                                    self.buffer.push(ch);
                                    self.last_ccc = Some(ch_class);
                                    continue;
                                }
                                match unicode::char::compose(k, ch) {
                                    Some(r) => {
                                        self.composee = Some(r);
                                        continue;
                                    }
                                    None => {
                                        self.buffer.push(ch);
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
/// The original string with all occurrences of `from` replaced with `to`
///
/// # Example
///
/// ```rust
/// use std::str;
/// let string = "orange";
/// let new_string = str::replace(string, "or", "str");
/// assert_eq!(new_string.as_slice(), "strange");
/// ```
pub fn replace(s: &str, from: &str, to: &str) -> String {
    let mut result = String::new();
    let mut last_end = 0;
    for (start, end) in s.match_indices(from) {
        result.push_str(unsafe{raw::slice_bytes(s, last_end, start)});
        result.push_str(to);
        last_end = end;
    }
    result.push_str(unsafe{raw::slice_bytes(s, last_end, s.len())});
    result
}

/*
Section: Misc
*/

/// Deprecated. Use `String::from_utf16`.
#[deprecated = "Replaced by String::from_utf16"]
pub fn from_utf16(v: &[u16]) -> Option<String> {
    String::from_utf16(v)
}

/// Deprecated. Use `String::from_utf16_lossy`.
#[deprecated = "Replaced by String::from_utf16_lossy"]
pub fn from_utf16_lossy(v: &[u16]) -> String {
    String::from_utf16_lossy(v)
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

/// Deprecated. Use `String::from_utf8_lossy`.
#[deprecated = "Replaced by String::from_utf8_lossy"]
pub fn from_utf8_lossy<'a>(v: &'a [u8]) -> MaybeOwned<'a> {
    String::from_utf8_lossy(v)
}

/*
Section: MaybeOwned
*/

/// A `MaybeOwned` is a string that can hold either a `String` or a `&str`.
/// This can be useful as an optimization when an allocation is sometimes
/// needed but not always.
pub enum MaybeOwned<'a> {
    /// A borrowed string
    Slice(&'a str),
    /// An owned string
    Owned(String)
}

/// `SendStr` is a specialization of `MaybeOwned` to be sendable
pub type SendStr = MaybeOwned<'static>;

impl<'a> MaybeOwned<'a> {
    /// Returns `true` if this `MaybeOwned` wraps an owned string
    ///
    /// # Example
    ///
    /// ```rust
    /// let string = String::from_str("orange");
    /// let maybe_owned_string = string.into_maybe_owned();
    /// assert_eq!(true, maybe_owned_string.is_owned());
    /// ```
    #[inline]
    pub fn is_owned(&self) -> bool {
        match *self {
            Slice(_) => false,
            Owned(_) => true
        }
    }

    /// Returns `true` if this `MaybeOwned` wraps a borrowed string
    ///
    /// # Example
    ///
    /// ```rust
    /// let string = "orange";
    /// let maybe_owned_string = string.as_slice().into_maybe_owned();
    /// assert_eq!(true, maybe_owned_string.is_slice());
    /// ```
    #[inline]
    pub fn is_slice(&self) -> bool {
        match *self {
            Slice(_) => true,
            Owned(_) => false
        }
    }
}

/// Trait for moving into a `MaybeOwned`
pub trait IntoMaybeOwned<'a> {
    /// Moves self into a `MaybeOwned`
    fn into_maybe_owned(self) -> MaybeOwned<'a>;
}

/// # Example
///
/// ```rust
/// let owned_string = String::from_str("orange");
/// let maybe_owned_string = owned_string.into_maybe_owned();
/// assert_eq!(true, maybe_owned_string.is_owned());
/// ```
impl<'a> IntoMaybeOwned<'a> for String {
    #[inline]
    fn into_maybe_owned(self) -> MaybeOwned<'a> {
        Owned(self)
    }
}

/// # Example
///
/// ```rust
/// let string = "orange";
/// let maybe_owned_str = string.as_slice().into_maybe_owned();
/// assert_eq!(false, maybe_owned_str.is_owned());
/// ```
impl<'a> IntoMaybeOwned<'a> for &'a str {
    #[inline]
    fn into_maybe_owned(self) -> MaybeOwned<'a> { Slice(self) }
}

/// # Example
///
/// ```rust
/// let str = "orange";
/// let maybe_owned_str = str.as_slice().into_maybe_owned();
/// let maybe_maybe_owned_str = maybe_owned_str.into_maybe_owned();
/// assert_eq!(false, maybe_maybe_owned_str.is_owned());
/// ```
impl<'a> IntoMaybeOwned<'a> for MaybeOwned<'a> {
    #[inline]
    fn into_maybe_owned(self) -> MaybeOwned<'a> { self }
}

impl<'a> PartialEq for MaybeOwned<'a> {
    #[inline]
    fn eq(&self, other: &MaybeOwned) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a> Eq for MaybeOwned<'a> {}

impl<'a> PartialOrd for MaybeOwned<'a> {
    #[inline]
    fn partial_cmp(&self, other: &MaybeOwned) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for MaybeOwned<'a> {
    #[inline]
    fn cmp(&self, other: &MaybeOwned) -> Ordering {
        self.as_slice().cmp(&other.as_slice())
    }
}

impl<'a, S: Str> Equiv<S> for MaybeOwned<'a> {
    #[inline]
    fn equiv(&self, other: &S) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a> Str for MaybeOwned<'a> {
    #[inline]
    fn as_slice<'b>(&'b self) -> &'b str {
        match *self {
            Slice(s) => s,
            Owned(ref s) => s.as_slice()
        }
    }
}

impl<'a> StrAllocating for MaybeOwned<'a> {
    #[inline]
    fn into_string(self) -> String {
        match self {
            Slice(s) => String::from_str(s),
            Owned(s) => s
        }
    }
}

impl<'a> Collection for MaybeOwned<'a> {
    #[inline]
    fn len(&self) -> uint { self.as_slice().len() }
}

impl<'a> Clone for MaybeOwned<'a> {
    #[inline]
    fn clone(&self) -> MaybeOwned<'a> {
        match *self {
            Slice(s) => Slice(s),
            Owned(ref s) => Owned(String::from_str(s.as_slice()))
        }
    }
}

impl<'a> Default for MaybeOwned<'a> {
    #[inline]
    fn default() -> MaybeOwned<'a> { Slice("") }
}

impl<'a, H: hash::Writer> hash::Hash<H> for MaybeOwned<'a> {
    #[inline]
    fn hash(&self, hasher: &mut H) {
        self.as_slice().hash(hasher)
    }
}

impl<'a> fmt::Show for MaybeOwned<'a> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Slice(ref s) => s.fmt(f),
            Owned(ref s) => s.fmt(f)
        }
    }
}

/// Unsafe operations
pub mod raw {
    use string;
    use string::String;
    use vec::Vec;

    use MutableSeq;

    pub use core::str::raw::{from_utf8, c_str_to_static_slice, slice_bytes};
    pub use core::str::raw::{slice_unchecked};

    /// Deprecated. Replaced by `string::raw::from_buf_len`
    #[deprecated = "Use string::raw::from_buf_len"]
    pub unsafe fn from_buf_len(buf: *const u8, len: uint) -> String {
        string::raw::from_buf_len(buf, len)
    }

    /// Deprecated. Use `string::raw::from_buf`
    #[deprecated = "Use string::raw::from_buf"]
    pub unsafe fn from_c_str(c_string: *const i8) -> String {
        string::raw::from_buf(c_string as *const u8)
    }

    /// Deprecated. Replaced by `string::raw::from_utf8`
    #[deprecated = "Use string::raw::from_utf8"]
    pub unsafe fn from_utf8_owned(v: Vec<u8>) -> String {
        string::raw::from_utf8(v)
    }

    /// Deprecated. Use `string::raw::from_utf8`
    #[deprecated = "Use string::raw::from_utf8"]
    pub unsafe fn from_byte(u: u8) -> String {
        string::raw::from_utf8(vec![u])
    }
}

/*
Section: Trait implementations
*/

/// Any string that can be represented as a slice
pub trait StrAllocating: Str {
    /// Convert `self` into a `String`, not making a copy if possible.
    fn into_string(self) -> String;

    #[allow(missing_doc)]
    #[deprecated = "replaced by .into_string()"]
    fn into_owned(self) -> String {
        self.into_string()
    }

    /// Escape each char in `s` with `char::escape_default`.
    fn escape_default(&self) -> String {
        let me = self.as_slice();
        let mut out = String::with_capacity(me.len());
        for c in me.chars() {
            c.escape_default(|c| out.push_char(c));
        }
        out
    }

    /// Escape each char in `s` with `char::escape_unicode`.
    fn escape_unicode(&self) -> String {
        let me = self.as_slice();
        let mut out = String::with_capacity(me.len());
        for c in me.chars() {
            c.escape_unicode(|c| out.push_char(c));
        }
        out
    }

    /// Replace all occurrences of one string with another.
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
    /// # Example
    ///
    /// ```rust
    /// let s = "Do you know the muffin man,
    /// The muffin man, the muffin man, ...".to_string();
    ///
    /// assert_eq!(s.replace("muffin man", "little lamb"),
    ///            "Do you know the little lamb,
    /// The little lamb, the little lamb, ...".to_string());
    ///
    /// // not found, so no change.
    /// assert_eq!(s.replace("cookie monster", "little lamb"), s);
    /// ```
    fn replace(&self, from: &str, to: &str) -> String {
        let me = self.as_slice();
        let mut result = String::new();
        let mut last_end = 0;
        for (start, end) in me.match_indices(from) {
            result.push_str(unsafe{raw::slice_bytes(me, last_end, start)});
            result.push_str(to);
            last_end = end;
        }
        result.push_str(unsafe{raw::slice_bytes(me, last_end, me.len())});
        result
    }

    #[allow(missing_doc)]
    #[deprecated = "obsolete, use `to_string`"]
    #[inline]
    fn to_owned(&self) -> String {
        unsafe {
            mem::transmute(Vec::from_slice(self.as_slice().as_bytes()))
        }
    }

    /// Converts to a vector of `u16` encoded as UTF-16.
    #[deprecated = "use `utf16_units` instead"]
    fn to_utf16(&self) -> Vec<u16> {
        self.as_slice().utf16_units().collect::<Vec<u16>>()
    }

    /// Given a string, make a new string with repeated copies of it.
    fn repeat(&self, nn: uint) -> String {
        let me = self.as_slice();
        let mut ret = String::with_capacity(nn * me.len());
        for _ in range(0, nn) {
            ret.push_str(me);
        }
        ret
    }

    /// Levenshtein Distance between two strings.
    fn lev_distance(&self, t: &str) -> uint {
        let me = self.as_slice();
        let slen = me.len();
        let tlen = t.len();

        if slen == 0 { return tlen; }
        if tlen == 0 { return slen; }

        let mut dcol = Vec::from_fn(tlen + 1, |x| x);

        for (i, sc) in me.chars().enumerate() {

            let mut current = i;
            *dcol.get_mut(0) = current + 1;

            for (j, tc) in t.chars().enumerate() {

                let next = dcol[j + 1];

                if sc == tc {
                    *dcol.get_mut(j + 1) = current;
                } else {
                    *dcol.get_mut(j + 1) = cmp::min(current, next);
                    *dcol.get_mut(j + 1) = cmp::min(dcol[j + 1],
                                                    dcol[j]) + 1;
                }

                current = next;
            }
        }

        return dcol[tlen];
    }

    /// An Iterator over the string in Unicode Normalization Form D
    /// (canonical decomposition).
    #[inline]
    fn nfd_chars<'a>(&'a self) -> Decompositions<'a> {
        Decompositions {
            iter: self.as_slice().chars(),
            buffer: Vec::new(),
            sorted: false,
            kind: Canonical
        }
    }

    /// An Iterator over the string in Unicode Normalization Form KD
    /// (compatibility decomposition).
    #[inline]
    fn nfkd_chars<'a>(&'a self) -> Decompositions<'a> {
        Decompositions {
            iter: self.as_slice().chars(),
            buffer: Vec::new(),
            sorted: false,
            kind: Compatible
        }
    }

    /// An Iterator over the string in Unicode Normalization Form C
    /// (canonical decomposition followed by canonical composition).
    #[inline]
    fn nfc_chars<'a>(&'a self) -> Recompositions<'a> {
        Recompositions {
            iter: self.nfd_chars(),
            state: Composing,
            buffer: RingBuf::new(),
            composee: None,
            last_ccc: None
        }
    }

    /// An Iterator over the string in Unicode Normalization Form KC
    /// (compatibility decomposition followed by canonical composition).
    #[inline]
    fn nfkc_chars<'a>(&'a self) -> Recompositions<'a> {
        Recompositions {
            iter: self.nfkd_chars(),
            state: Composing,
            buffer: RingBuf::new(),
            composee: None,
            last_ccc: None
        }
    }
}

impl<'a> StrAllocating for &'a str {
    #[inline]
    fn into_string(self) -> String {
        String::from_str(self)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::AdditiveIterator;
    use std::iter::range;
    use std::default::Default;
    use std::char::Char;
    use std::clone::Clone;
    use std::cmp::{Equal, Greater, Less, Ord, PartialOrd, Equiv};
    use std::option::{Some, None};
    use std::ptr::RawPtr;
    use std::iter::{Iterator, DoubleEndedIterator};
    use {Collection, MutableSeq};

    use super::*;
    use std::slice::{Slice, ImmutableSlice};
    use string::String;
    use vec::Vec;

    use unicode::char::UnicodeChar;

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

        assert_eq!("ｈｅｌｌｏ".width(false), 10u);
        assert_eq!("ｈｅｌｌｏ".width(true), 10u);
        assert_eq!("\0\0\0\0\0".width(false), 0u);
        assert_eq!("\0\0\0\0\0".width(true), 0u);
        assert_eq!("".width(false), 0u);
        assert_eq!("".width(true), 0u);
        assert_eq!("\u2081\u2082\u2083\u2084".width(false), 4u);
        assert_eq!("\u2081\u2082\u2083\u2084".width(true), 8u);
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
    fn test_collect() {
        let empty = String::from_str("");
        let s: String = empty.as_slice().chars().collect();
        assert_eq!(empty, s);
        let data = String::from_str("ประเทศไทย中");
        let s: String = data.as_slice().chars().collect();
        assert_eq!(data, s);
    }

    #[test]
    fn test_into_bytes() {
        let data = String::from_str("asdf");
        let buf = data.into_bytes();
        assert_eq!(b"asdf", buf.as_slice());
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

        let string = "ประเทศไทย中华Việt Nam";
        let mut data = String::from_str(string);
        data.push_str(string);
        assert!(data.as_slice().find_str("ไท华").is_none());
        assert_eq!(data.as_slice().slice(0u, 43u).find_str(""), Some(0u));
        assert_eq!(data.as_slice().slice(6u, 43u).find_str(""), Some(6u - 6u));

        assert_eq!(data.as_slice().slice(0u, 43u).find_str("ประ"), Some( 0u));
        assert_eq!(data.as_slice().slice(0u, 43u).find_str("ทศไ"), Some(12u));
        assert_eq!(data.as_slice().slice(0u, 43u).find_str("ย中"), Some(24u));
        assert_eq!(data.as_slice().slice(0u, 43u).find_str("iệt"), Some(34u));
        assert_eq!(data.as_slice().slice(0u, 43u).find_str("Nam"), Some(40u));

        assert_eq!(data.as_slice().slice(43u, 86u).find_str("ประ"), Some(43u - 43u));
        assert_eq!(data.as_slice().slice(43u, 86u).find_str("ทศไ"), Some(55u - 43u));
        assert_eq!(data.as_slice().slice(43u, 86u).find_str("ย中"), Some(67u - 43u));
        assert_eq!(data.as_slice().slice(43u, 86u).find_str("iệt"), Some(77u - 43u));
        assert_eq!(data.as_slice().slice(43u, 86u).find_str("Nam"), Some(83u - 43u));
    }

    #[test]
    fn test_slice_chars() {
        fn t(a: &str, b: &str, start: uint) {
            assert_eq!(a.slice_chars(start, start + b.char_len()), b);
        }
        t("", "", 0);
        t("hello", "llo", 2);
        t("hello", "el", 1);
        t("αβλ", "β", 1);
        t("αβλ", "", 3);
        assert_eq!("ะเทศไท", "ประเทศไทย中华Việt Nam".slice_chars(2, 8));
    }

    #[test]
    fn test_concat() {
        fn t(v: &[String], s: &str) {
            assert_eq!(v.concat().as_slice(), s);
        }
        t([String::from_str("you"), String::from_str("know"),
           String::from_str("I'm"),
           String::from_str("no"), String::from_str("good")],
          "youknowI'mnogood");
        let v: &[String] = [];
        t(v, "");
        t([String::from_str("hi")], "hi");
    }

    #[test]
    fn test_connect() {
        fn t(v: &[String], sep: &str, s: &str) {
            assert_eq!(v.connect(sep).as_slice(), s);
        }
        t([String::from_str("you"), String::from_str("know"),
           String::from_str("I'm"),
           String::from_str("no"), String::from_str("good")],
          " ", "you know I'm no good");
        let v: &[String] = [];
        t(v, " ", "");
        t([String::from_str("hi")], " ", "hi");
    }

    #[test]
    fn test_concat_slices() {
        fn t(v: &[&str], s: &str) {
            assert_eq!(v.concat().as_slice(), s);
        }
        t(["you", "know", "I'm", "no", "good"], "youknowI'mnogood");
        let v: &[&str] = [];
        t(v, "");
        t(["hi"], "hi");
    }

    #[test]
    fn test_connect_slices() {
        fn t(v: &[&str], sep: &str, s: &str) {
            assert_eq!(v.connect(sep).as_slice(), s);
        }
        t(["you", "know", "I'm", "no", "good"],
          " ", "you know I'm no good");
        t([], " ", "");
        t(["hi"], " ", "hi");
    }

    #[test]
    fn test_repeat() {
        assert_eq!("x".repeat(4), String::from_str("xxxx"));
        assert_eq!("hi".repeat(4), String::from_str("hihihihi"));
        assert_eq!("ไท华".repeat(3), String::from_str("ไท华ไท华ไท华"));
        assert_eq!("".repeat(4), String::from_str(""));
        assert_eq!("hi".repeat(0), String::from_str(""));
    }

    #[test]
    fn test_unsafe_slice() {
        assert_eq!("ab", unsafe {raw::slice_bytes("abc", 0, 2)});
        assert_eq!("bc", unsafe {raw::slice_bytes("abc", 1, 3)});
        assert_eq!("", unsafe {raw::slice_bytes("abc", 1, 1)});
        fn a_million_letter_a() -> String {
            let mut i = 0u;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("aaaaaaaaaa");
                i += 1;
            }
            rs
        }
        fn half_a_million_letter_a() -> String {
            let mut i = 0u;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("aaaaa");
                i += 1;
            }
            rs
        }
        let letters = a_million_letter_a();
        assert!(half_a_million_letter_a() ==
            unsafe {String::from_str(raw::slice_bytes(letters.as_slice(),
                                     0u,
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
        assert_eq!(data.replace(a, repl).as_slice(), a2);
    }

    #[test]
    fn test_replace_2b() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let b = "ะเ";
        let b2 = "ปรدولة الكويتทศไทย中华";
        assert_eq!(data.replace(b, repl).as_slice(), b2);
    }

    #[test]
    fn test_replace_2c() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let c = "中华";
        let c2 = "ประเทศไทยدولة الكويت";
        assert_eq!(data.replace(c, repl).as_slice(), c2);
    }

    #[test]
    fn test_replace_2d() {
        let data = "ประเทศไทย中华";
        let repl = "دولة الكويت";

        let d = "ไท华";
        assert_eq!(data.replace(d, repl).as_slice(), data);
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

        fn a_million_letter_x() -> String {
            let mut i = 0u;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("华华华华华华华华华华");
                i += 1;
            }
            rs
        }
        fn half_a_million_letter_x() -> String {
            let mut i = 0u;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("华华华华华");
                i += 1;
            }
            rs
        }
        let letters = a_million_letter_x();
        assert!(half_a_million_letter_x() ==
            String::from_str(letters.as_slice().slice(0u, 3u * 500000u)));
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
        assert_eq!(" *** foo *** ".trim_left_chars(v), " *** foo *** ");
        assert_eq!(" *** foo *** ".trim_left_chars(&['*', ' ']), "foo *** ");
        assert_eq!(" ***  *** ".trim_left_chars(&['*', ' ']), "");
        assert_eq!("foo *** ".trim_left_chars(&['*', ' ']), "foo *** ");

        assert_eq!("11foo1bar11".trim_left_chars('1'), "foo1bar11");
        assert_eq!("12foo1bar12".trim_left_chars(&['1', '2']), "foo1bar12");
        assert_eq!("123foo1bar123".trim_left_chars(|c: char| c.is_digit()), "foo1bar123");
    }

    #[test]
    fn test_trim_right_chars() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_right_chars(v), " *** foo *** ");
        assert_eq!(" *** foo *** ".trim_right_chars(&['*', ' ']), " *** foo");
        assert_eq!(" ***  *** ".trim_right_chars(&['*', ' ']), "");
        assert_eq!(" *** foo".trim_right_chars(&['*', ' ']), " *** foo");

        assert_eq!("11foo1bar11".trim_right_chars('1'), "11foo1bar");
        assert_eq!("12foo1bar12".trim_right_chars(&['1', '2']), "12foo1bar");
        assert_eq!("123foo1bar123".trim_right_chars(|c: char| c.is_digit()), "123foo1bar");
    }

    #[test]
    fn test_trim_chars() {
        let v: &[char] = &[];
        assert_eq!(" *** foo *** ".trim_chars(v), " *** foo *** ");
        assert_eq!(" *** foo *** ".trim_chars(&['*', ' ']), "foo");
        assert_eq!(" ***  *** ".trim_chars(&['*', ' ']), "");
        assert_eq!("foo".trim_chars(&['*', ' ']), "foo");

        assert_eq!("11foo1bar11".trim_chars('1'), "foo1bar");
        assert_eq!("12foo1bar12".trim_chars(&['1', '2']), "foo1bar");
        assert_eq!("123foo1bar123".trim_chars(|c: char| c.is_digit()), "foo1bar");
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
    fn test_slice_shift_char() {
        let data = "ประเทศไทย中";
        assert_eq!(data.slice_shift_char(), (Some('ป'), "ระเทศไทย中"));
    }

    #[test]
    fn test_slice_shift_char_2() {
        let empty = "";
        assert_eq!(empty.slice_shift_char(), (None, ""));
    }

    #[test]
    fn test_is_utf8() {
        // deny overlong encodings
        assert!(!is_utf8([0xc0, 0x80]));
        assert!(!is_utf8([0xc0, 0xae]));
        assert!(!is_utf8([0xe0, 0x80, 0x80]));
        assert!(!is_utf8([0xe0, 0x80, 0xaf]));
        assert!(!is_utf8([0xe0, 0x81, 0x81]));
        assert!(!is_utf8([0xf0, 0x82, 0x82, 0xac]));
        assert!(!is_utf8([0xf4, 0x90, 0x80, 0x80]));

        // deny surrogates
        assert!(!is_utf8([0xED, 0xA0, 0x80]));
        assert!(!is_utf8([0xED, 0xBF, 0xBF]));

        assert!(is_utf8([0xC2, 0x80]));
        assert!(is_utf8([0xDF, 0xBF]));
        assert!(is_utf8([0xE0, 0xA0, 0x80]));
        assert!(is_utf8([0xED, 0x9F, 0xBF]));
        assert!(is_utf8([0xEE, 0x80, 0x80]));
        assert!(is_utf8([0xEF, 0xBF, 0xBF]));
        assert!(is_utf8([0xF0, 0x90, 0x80, 0x80]));
        assert!(is_utf8([0xF4, 0x8F, 0xBF, 0xBF]));
    }

    #[test]
    fn test_is_utf16() {
        macro_rules! pos ( ($($e:expr),*) => { { $(assert!(is_utf16($e));)* } });

        // non-surrogates
        pos!([0x0000],
             [0x0001, 0x0002],
             [0xD7FF],
             [0xE000]);

        // surrogate pairs (randomly generated with Python 3's
        // .encode('utf-16be'))
        pos!([0xdb54, 0xdf16, 0xd880, 0xdee0, 0xdb6a, 0xdd45],
             [0xd91f, 0xdeb1, 0xdb31, 0xdd84, 0xd8e2, 0xde14],
             [0xdb9f, 0xdc26, 0xdb6f, 0xde58, 0xd850, 0xdfae]);

        // mixtures (also random)
        pos!([0xd921, 0xdcc2, 0x002d, 0x004d, 0xdb32, 0xdf65],
             [0xdb45, 0xdd2d, 0x006a, 0xdacd, 0xddfe, 0x0006],
             [0x0067, 0xd8ff, 0xddb7, 0x000f, 0xd900, 0xdc80]);

        // negative tests
        macro_rules! neg ( ($($e:expr),*) => { { $(assert!(!is_utf16($e));)* } });

        neg!(
            // surrogate + regular unit
            [0xdb45, 0x0000],
            // surrogate + lead surrogate
            [0xd900, 0xd900],
            // unterminated surrogate
            [0xd8ff],
            // trail surrogate without a lead
            [0xddb7]);

        // random byte sequences that Python 3's .decode('utf-16be')
        // failed on
        neg!([0x5b3d, 0x0141, 0xde9e, 0x8fdc, 0xc6e7],
             [0xdf5a, 0x82a5, 0x62b9, 0xb447, 0x92f3],
             [0xda4e, 0x42bc, 0x4462, 0xee98, 0xc2ca],
             [0xbe00, 0xb04a, 0x6ecb, 0xdd89, 0xe278],
             [0x0465, 0xab56, 0xdbb6, 0xa893, 0x665e],
             [0x6b7f, 0x0a19, 0x40f4, 0xa657, 0xdcc5],
             [0x9b50, 0xda5e, 0x24ec, 0x03ad, 0x6dee],
             [0x8d17, 0xcaa7, 0xf4ae, 0xdf6e, 0xbed7],
             [0xdaee, 0x2584, 0x7d30, 0xa626, 0x121a],
             [0xd956, 0x4b43, 0x7570, 0xccd6, 0x4f4a],
             [0x9dcf, 0x1b49, 0x4ba5, 0xfce9, 0xdffe],
             [0x6572, 0xce53, 0xb05a, 0xf6af, 0xdacf],
             [0x1b90, 0x728c, 0x9906, 0xdb68, 0xf46e],
             [0x1606, 0xbeca, 0xbe76, 0x860f, 0xdfa5],
             [0x8b4f, 0xde7a, 0xd220, 0x9fac, 0x2b6f],
             [0xb8fe, 0xebbe, 0xda32, 0x1a5f, 0x8b8b],
             [0x934b, 0x8956, 0xc434, 0x1881, 0xddf7],
             [0x5a95, 0x13fc, 0xf116, 0xd89b, 0x93f9],
             [0xd640, 0x71f1, 0xdd7d, 0x77eb, 0x1cd8],
             [0x348b, 0xaef0, 0xdb2c, 0xebf1, 0x1282],
             [0x50d7, 0xd824, 0x5010, 0xb369, 0x22ea]);
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
        assert_eq!("abc".as_bytes(), b"abc");
        assert_eq!("ศไทย中华Việt Nam".as_bytes(), v.as_slice());
    }

    #[test]
    #[should_fail]
    fn test_as_bytes_fail() {
        // Don't double free. (I'm not sure if this exercises the
        // original problem code path anymore.)
        let s = String::from_str("");
        let _bytes = s.as_bytes();
        fail!();
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
        let b = a.slice(7, a.len());
        let c = a.slice(0, a.len() - 6);
        assert_eq!(a.subslice_offset(b), 7);
        assert_eq!(a.subslice_offset(c), 0);

        let string = "a\nb\nc";
        let lines: Vec<&str> = string.lines().collect();
        let lines = lines.as_slice();
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

        let v: Vec<u8> = Vec::from_slice(s1.as_bytes());
        let s2: String = String::from_str(from_utf8(v.as_slice()).unwrap());
        let mut i: uint = 0u;
        let n1: uint = s1.len();
        let n2: uint = v.len();
        assert_eq!(n1, n2);
        while i < n1 {
            let a: u8 = s1.as_bytes()[i];
            let b: u8 = s2.as_bytes()[i];
            debug!("{}", a);
            debug!("{}", b);
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
    fn test_truncate_utf16_at_nul() {
        let v = [];
        assert_eq!(truncate_utf16_at_nul(v), &[]);

        let v = [0, 2, 3];
        assert_eq!(truncate_utf16_at_nul(v), &[]);

        let v = [1, 0, 3];
        assert_eq!(truncate_utf16_at_nul(v), &[1]);

        let v = [1, 2, 0];
        assert_eq!(truncate_utf16_at_nul(v), &[1, 2]);

        let v = [1, 2, 3];
        assert_eq!(truncate_utf16_at_nul(v), &[1, 2, 3]);
    }

    #[test]
    fn test_char_at() {
        let s = "ศไทย中华Việt Nam";
        let v = vec!['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
        let mut pos = 0;
        for ch in v.iter() {
            assert!(s.char_at(pos) == *ch);
            pos += String::from_char(1, *ch).len();
        }
    }

    #[test]
    fn test_char_at_reverse() {
        let s = "ศไทย中华Việt Nam";
        let v = vec!['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];
        let mut pos = s.len();
        for ch in v.iter().rev() {
            assert!(s.char_at_reverse(pos) == *ch);
            pos -= String::from_char(1, *ch).len();
        }
    }

    #[test]
    fn test_escape_unicode() {
        assert_eq!("abc".escape_unicode(), String::from_str("\\x61\\x62\\x63"));
        assert_eq!("a c".escape_unicode(), String::from_str("\\x61\\x20\\x63"));
        assert_eq!("\r\n\t".escape_unicode(), String::from_str("\\x0d\\x0a\\x09"));
        assert_eq!("'\"\\".escape_unicode(), String::from_str("\\x27\\x22\\x5c"));
        assert_eq!("\x00\x01\xfe\xff".escape_unicode(), String::from_str("\\x00\\x01\\xfe\\xff"));
        assert_eq!("\u0100\uffff".escape_unicode(), String::from_str("\\u0100\\uffff"));
        assert_eq!("\U00010000\U0010ffff".escape_unicode(),
                   String::from_str("\\U00010000\\U0010ffff"));
        assert_eq!("ab\ufb00".escape_unicode(), String::from_str("\\x61\\x62\\ufb00"));
        assert_eq!("\U0001d4ea\r".escape_unicode(), String::from_str("\\U0001d4ea\\x0d"));
    }

    #[test]
    fn test_escape_default() {
        assert_eq!("abc".escape_default(), String::from_str("abc"));
        assert_eq!("a c".escape_default(), String::from_str("a c"));
        assert_eq!("\r\n\t".escape_default(), String::from_str("\\r\\n\\t"));
        assert_eq!("'\"\\".escape_default(), String::from_str("\\'\\\"\\\\"));
        assert_eq!("\u0100\uffff".escape_default(), String::from_str("\\u0100\\uffff"));
        assert_eq!("\U00010000\U0010ffff".escape_default(),
                   String::from_str("\\U00010000\\U0010ffff"));
        assert_eq!("ab\ufb00".escape_default(), String::from_str("ab\\ufb00"));
        assert_eq!("\U0001d4ea\r".escape_default(), String::from_str("\\U0001d4ea\\r"));
    }

    #[test]
    fn test_total_ord() {
        "1234".cmp(&("123")) == Greater;
        "123".cmp(&("1234")) == Less;
        "1234".cmp(&("1234")) == Equal;
        "12345555".cmp(&("123456")) == Less;
        "22".cmp(&("1234")) == Greater;
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
        let mut it = s.chars();

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
        let mut it = s.chars().rev();

        for c in it {
            assert_eq!(c, v[pos]);
            pos += 1;
        }
        assert_eq!(pos, v.len());
    }

    #[test]
    fn test_chars_decoding() {
        let mut bytes = [0u8, ..4];
        for c in range(0u32, 0x110000).filter_map(|c| ::core::char::from_u32(c)) {
            let len = c.encode_utf8(bytes).unwrap_or(0);
            let s = ::core::str::from_utf8(bytes.slice_to(len)).unwrap();
            if Some(c) != s.chars().next() {
                fail!("character {:x}={} does not decode correctly", c as u32, c);
            }
        }
    }

    #[test]
    fn test_chars_rev_decoding() {
        let mut bytes = [0u8, ..4];
        for c in range(0u32, 0x110000).filter_map(|c| ::core::char::from_u32(c)) {
            let len = c.encode_utf8(bytes).unwrap_or(0);
            let s = ::core::str::from_utf8(bytes.slice_to(len)).unwrap();
            if Some(c) != s.chars().rev().next() {
                fail!("character {:x}={} does not decode correctly", c as u32, c);
            }
        }
    }

    #[test]
    fn test_iterator_clone() {
        let s = "ศไทย中华Việt Nam";
        let mut it = s.chars();
        it.next();
        assert!(it.zip(it.clone()).all(|(x,y)| x == y));
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
        let mut it = s.char_indices();

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
        let mut it = s.char_indices().rev();

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

        let split: Vec<&str> = data.split(' ').collect();
        assert_eq!( split, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let mut rsplit: Vec<&str> = data.split(' ').rev().collect();
        rsplit.reverse();
        assert_eq!(rsplit, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let split: Vec<&str> = data.split(|c: char| c == ' ').collect();
        assert_eq!( split, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let mut rsplit: Vec<&str> = data.split(|c: char| c == ' ').rev().collect();
        rsplit.reverse();
        assert_eq!(rsplit, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        // Unicode
        let split: Vec<&str> = data.split('ä').collect();
        assert_eq!( split, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let mut rsplit: Vec<&str> = data.split('ä').rev().collect();
        rsplit.reverse();
        assert_eq!(rsplit, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let split: Vec<&str> = data.split(|c: char| c == 'ä').collect();
        assert_eq!( split, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let mut rsplit: Vec<&str> = data.split(|c: char| c == 'ä').rev().collect();
        rsplit.reverse();
        assert_eq!(rsplit, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);
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
    fn test_rsplitn_char_iterator() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let mut split: Vec<&str> = data.rsplitn(3, ' ').collect();
        split.reverse();
        assert_eq!(split, vec!["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

        let mut split: Vec<&str> = data.rsplitn(3, |c: char| c == ' ').collect();
        split.reverse();
        assert_eq!(split, vec!["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

        // Unicode
        let mut split: Vec<&str> = data.rsplitn(3, 'ä').collect();
        split.reverse();
        assert_eq!(split, vec!["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);

        let mut split: Vec<&str> = data.rsplitn(3, |c: char| c == 'ä').collect();
        split.reverse();
        assert_eq!(split, vec!["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);
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
    fn test_rev_split_char_iterator_no_trailing() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let mut split: Vec<&str> = data.split('\n').rev().collect();
        split.reverse();
        assert_eq!(split, vec!["", "Märy häd ä little lämb", "Little lämb", ""]);

        let mut split: Vec<&str> = data.split_terminator('\n').rev().collect();
        split.reverse();
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
                assert_eq!($input.nfd_chars().collect::<String>(), $expected.into_string());
            }
        }
        t!("abc", "abc");
        t!("\u1e0b\u01c4", "d\u0307\u01c4");
        t!("\u2026", "\u2026");
        t!("\u2126", "\u03a9");
        t!("\u1e0b\u0323", "d\u0323\u0307");
        t!("\u1e0d\u0307", "d\u0323\u0307");
        t!("a\u0301", "a\u0301");
        t!("\u0301a", "\u0301a");
        t!("\ud4db", "\u1111\u1171\u11b6");
        t!("\uac1c", "\u1100\u1162");
    }

    #[test]
    fn test_nfkd_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfkd_chars().collect::<String>(), $expected.into_string());
            }
        }
        t!("abc", "abc");
        t!("\u1e0b\u01c4", "d\u0307DZ\u030c");
        t!("\u2026", "...");
        t!("\u2126", "\u03a9");
        t!("\u1e0b\u0323", "d\u0323\u0307");
        t!("\u1e0d\u0307", "d\u0323\u0307");
        t!("a\u0301", "a\u0301");
        t!("\u0301a", "\u0301a");
        t!("\ud4db", "\u1111\u1171\u11b6");
        t!("\uac1c", "\u1100\u1162");
    }

    #[test]
    fn test_nfc_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfc_chars().collect::<String>(), $expected.into_string());
            }
        }
        t!("abc", "abc");
        t!("\u1e0b\u01c4", "\u1e0b\u01c4");
        t!("\u2026", "\u2026");
        t!("\u2126", "\u03a9");
        t!("\u1e0b\u0323", "\u1e0d\u0307");
        t!("\u1e0d\u0307", "\u1e0d\u0307");
        t!("a\u0301", "\xe1");
        t!("\u0301a", "\u0301a");
        t!("\ud4db", "\ud4db");
        t!("\uac1c", "\uac1c");
        t!("a\u0300\u0305\u0315\u05aeb", "\xe0\u05ae\u0305\u0315b");
    }

    #[test]
    fn test_nfkc_chars() {
        macro_rules! t {
            ($input: expr, $expected: expr) => {
                assert_eq!($input.nfkc_chars().collect::<String>(), $expected.into_string());
            }
        }
        t!("abc", "abc");
        t!("\u1e0b\u01c4", "\u1e0bD\u017d");
        t!("\u2026", "...");
        t!("\u2126", "\u03a9");
        t!("\u1e0b\u0323", "\u1e0d\u0307");
        t!("\u1e0d\u0307", "\u1e0d\u0307");
        t!("a\u0301", "\xe1");
        t!("\u0301a", "\u0301a");
        t!("\ud4db", "\ud4db");
        t!("\uac1c", "\uac1c");
        t!("a\u0300\u0305\u0315\u05aeb", "\xe0\u05ae\u0305\u0315b");
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
        use std::iter::order;
        // official Unicode test data
        // from http://www.unicode.org/Public/UCD/latest/ucd/auxiliary/GraphemeBreakTest.txt
        let test_same = [
            ("\u0020\u0020", &["\u0020", "\u0020"]), ("\u0020\u0308\u0020", &["\u0020\u0308",
            "\u0020"]), ("\u0020\u000D", &["\u0020", "\u000D"]), ("\u0020\u0308\u000D",
            &["\u0020\u0308", "\u000D"]), ("\u0020\u000A", &["\u0020", "\u000A"]),
            ("\u0020\u0308\u000A", &["\u0020\u0308", "\u000A"]), ("\u0020\u0001", &["\u0020",
            "\u0001"]), ("\u0020\u0308\u0001", &["\u0020\u0308", "\u0001"]), ("\u0020\u0300",
            &["\u0020\u0300"]), ("\u0020\u0308\u0300", &["\u0020\u0308\u0300"]), ("\u0020\u1100",
            &["\u0020", "\u1100"]), ("\u0020\u0308\u1100", &["\u0020\u0308", "\u1100"]),
            ("\u0020\u1160", &["\u0020", "\u1160"]), ("\u0020\u0308\u1160", &["\u0020\u0308",
            "\u1160"]), ("\u0020\u11A8", &["\u0020", "\u11A8"]), ("\u0020\u0308\u11A8",
            &["\u0020\u0308", "\u11A8"]), ("\u0020\uAC00", &["\u0020", "\uAC00"]),
            ("\u0020\u0308\uAC00", &["\u0020\u0308", "\uAC00"]), ("\u0020\uAC01", &["\u0020",
            "\uAC01"]), ("\u0020\u0308\uAC01", &["\u0020\u0308", "\uAC01"]), ("\u0020\U0001F1E6",
            &["\u0020", "\U0001F1E6"]), ("\u0020\u0308\U0001F1E6", &["\u0020\u0308",
            "\U0001F1E6"]), ("\u0020\u0378", &["\u0020", "\u0378"]), ("\u0020\u0308\u0378",
            &["\u0020\u0308", "\u0378"]), ("\u000D\u0020", &["\u000D", "\u0020"]),
            ("\u000D\u0308\u0020", &["\u000D", "\u0308", "\u0020"]), ("\u000D\u000D", &["\u000D",
            "\u000D"]), ("\u000D\u0308\u000D", &["\u000D", "\u0308", "\u000D"]), ("\u000D\u000A",
            &["\u000D\u000A"]), ("\u000D\u0308\u000A", &["\u000D", "\u0308", "\u000A"]),
            ("\u000D\u0001", &["\u000D", "\u0001"]), ("\u000D\u0308\u0001", &["\u000D", "\u0308",
            "\u0001"]), ("\u000D\u0300", &["\u000D", "\u0300"]), ("\u000D\u0308\u0300",
            &["\u000D", "\u0308\u0300"]), ("\u000D\u0903", &["\u000D", "\u0903"]),
            ("\u000D\u1100", &["\u000D", "\u1100"]), ("\u000D\u0308\u1100", &["\u000D", "\u0308",
            "\u1100"]), ("\u000D\u1160", &["\u000D", "\u1160"]), ("\u000D\u0308\u1160",
            &["\u000D", "\u0308", "\u1160"]), ("\u000D\u11A8", &["\u000D", "\u11A8"]),
            ("\u000D\u0308\u11A8", &["\u000D", "\u0308", "\u11A8"]), ("\u000D\uAC00", &["\u000D",
            "\uAC00"]), ("\u000D\u0308\uAC00", &["\u000D", "\u0308", "\uAC00"]), ("\u000D\uAC01",
            &["\u000D", "\uAC01"]), ("\u000D\u0308\uAC01", &["\u000D", "\u0308", "\uAC01"]),
            ("\u000D\U0001F1E6", &["\u000D", "\U0001F1E6"]), ("\u000D\u0308\U0001F1E6",
            &["\u000D", "\u0308", "\U0001F1E6"]), ("\u000D\u0378", &["\u000D", "\u0378"]),
            ("\u000D\u0308\u0378", &["\u000D", "\u0308", "\u0378"]), ("\u000A\u0020", &["\u000A",
            "\u0020"]), ("\u000A\u0308\u0020", &["\u000A", "\u0308", "\u0020"]), ("\u000A\u000D",
            &["\u000A", "\u000D"]), ("\u000A\u0308\u000D", &["\u000A", "\u0308", "\u000D"]),
            ("\u000A\u000A", &["\u000A", "\u000A"]), ("\u000A\u0308\u000A", &["\u000A", "\u0308",
            "\u000A"]), ("\u000A\u0001", &["\u000A", "\u0001"]), ("\u000A\u0308\u0001",
            &["\u000A", "\u0308", "\u0001"]), ("\u000A\u0300", &["\u000A", "\u0300"]),
            ("\u000A\u0308\u0300", &["\u000A", "\u0308\u0300"]), ("\u000A\u0903", &["\u000A",
            "\u0903"]), ("\u000A\u1100", &["\u000A", "\u1100"]), ("\u000A\u0308\u1100",
            &["\u000A", "\u0308", "\u1100"]), ("\u000A\u1160", &["\u000A", "\u1160"]),
            ("\u000A\u0308\u1160", &["\u000A", "\u0308", "\u1160"]), ("\u000A\u11A8", &["\u000A",
            "\u11A8"]), ("\u000A\u0308\u11A8", &["\u000A", "\u0308", "\u11A8"]), ("\u000A\uAC00",
            &["\u000A", "\uAC00"]), ("\u000A\u0308\uAC00", &["\u000A", "\u0308", "\uAC00"]),
            ("\u000A\uAC01", &["\u000A", "\uAC01"]), ("\u000A\u0308\uAC01", &["\u000A", "\u0308",
            "\uAC01"]), ("\u000A\U0001F1E6", &["\u000A", "\U0001F1E6"]),
            ("\u000A\u0308\U0001F1E6", &["\u000A", "\u0308", "\U0001F1E6"]), ("\u000A\u0378",
            &["\u000A", "\u0378"]), ("\u000A\u0308\u0378", &["\u000A", "\u0308", "\u0378"]),
            ("\u0001\u0020", &["\u0001", "\u0020"]), ("\u0001\u0308\u0020", &["\u0001", "\u0308",
            "\u0020"]), ("\u0001\u000D", &["\u0001", "\u000D"]), ("\u0001\u0308\u000D",
            &["\u0001", "\u0308", "\u000D"]), ("\u0001\u000A", &["\u0001", "\u000A"]),
            ("\u0001\u0308\u000A", &["\u0001", "\u0308", "\u000A"]), ("\u0001\u0001", &["\u0001",
            "\u0001"]), ("\u0001\u0308\u0001", &["\u0001", "\u0308", "\u0001"]), ("\u0001\u0300",
            &["\u0001", "\u0300"]), ("\u0001\u0308\u0300", &["\u0001", "\u0308\u0300"]),
            ("\u0001\u0903", &["\u0001", "\u0903"]), ("\u0001\u1100", &["\u0001", "\u1100"]),
            ("\u0001\u0308\u1100", &["\u0001", "\u0308", "\u1100"]), ("\u0001\u1160", &["\u0001",
            "\u1160"]), ("\u0001\u0308\u1160", &["\u0001", "\u0308", "\u1160"]), ("\u0001\u11A8",
            &["\u0001", "\u11A8"]), ("\u0001\u0308\u11A8", &["\u0001", "\u0308", "\u11A8"]),
            ("\u0001\uAC00", &["\u0001", "\uAC00"]), ("\u0001\u0308\uAC00", &["\u0001", "\u0308",
            "\uAC00"]), ("\u0001\uAC01", &["\u0001", "\uAC01"]), ("\u0001\u0308\uAC01",
            &["\u0001", "\u0308", "\uAC01"]), ("\u0001\U0001F1E6", &["\u0001", "\U0001F1E6"]),
            ("\u0001\u0308\U0001F1E6", &["\u0001", "\u0308", "\U0001F1E6"]), ("\u0001\u0378",
            &["\u0001", "\u0378"]), ("\u0001\u0308\u0378", &["\u0001", "\u0308", "\u0378"]),
            ("\u0300\u0020", &["\u0300", "\u0020"]), ("\u0300\u0308\u0020", &["\u0300\u0308",
            "\u0020"]), ("\u0300\u000D", &["\u0300", "\u000D"]), ("\u0300\u0308\u000D",
            &["\u0300\u0308", "\u000D"]), ("\u0300\u000A", &["\u0300", "\u000A"]),
            ("\u0300\u0308\u000A", &["\u0300\u0308", "\u000A"]), ("\u0300\u0001", &["\u0300",
            "\u0001"]), ("\u0300\u0308\u0001", &["\u0300\u0308", "\u0001"]), ("\u0300\u0300",
            &["\u0300\u0300"]), ("\u0300\u0308\u0300", &["\u0300\u0308\u0300"]), ("\u0300\u1100",
            &["\u0300", "\u1100"]), ("\u0300\u0308\u1100", &["\u0300\u0308", "\u1100"]),
            ("\u0300\u1160", &["\u0300", "\u1160"]), ("\u0300\u0308\u1160", &["\u0300\u0308",
            "\u1160"]), ("\u0300\u11A8", &["\u0300", "\u11A8"]), ("\u0300\u0308\u11A8",
            &["\u0300\u0308", "\u11A8"]), ("\u0300\uAC00", &["\u0300", "\uAC00"]),
            ("\u0300\u0308\uAC00", &["\u0300\u0308", "\uAC00"]), ("\u0300\uAC01", &["\u0300",
            "\uAC01"]), ("\u0300\u0308\uAC01", &["\u0300\u0308", "\uAC01"]), ("\u0300\U0001F1E6",
            &["\u0300", "\U0001F1E6"]), ("\u0300\u0308\U0001F1E6", &["\u0300\u0308",
            "\U0001F1E6"]), ("\u0300\u0378", &["\u0300", "\u0378"]), ("\u0300\u0308\u0378",
            &["\u0300\u0308", "\u0378"]), ("\u0903\u0020", &["\u0903", "\u0020"]),
            ("\u0903\u0308\u0020", &["\u0903\u0308", "\u0020"]), ("\u0903\u000D", &["\u0903",
            "\u000D"]), ("\u0903\u0308\u000D", &["\u0903\u0308", "\u000D"]), ("\u0903\u000A",
            &["\u0903", "\u000A"]), ("\u0903\u0308\u000A", &["\u0903\u0308", "\u000A"]),
            ("\u0903\u0001", &["\u0903", "\u0001"]), ("\u0903\u0308\u0001", &["\u0903\u0308",
            "\u0001"]), ("\u0903\u0300", &["\u0903\u0300"]), ("\u0903\u0308\u0300",
            &["\u0903\u0308\u0300"]), ("\u0903\u1100", &["\u0903", "\u1100"]),
            ("\u0903\u0308\u1100", &["\u0903\u0308", "\u1100"]), ("\u0903\u1160", &["\u0903",
            "\u1160"]), ("\u0903\u0308\u1160", &["\u0903\u0308", "\u1160"]), ("\u0903\u11A8",
            &["\u0903", "\u11A8"]), ("\u0903\u0308\u11A8", &["\u0903\u0308", "\u11A8"]),
            ("\u0903\uAC00", &["\u0903", "\uAC00"]), ("\u0903\u0308\uAC00", &["\u0903\u0308",
            "\uAC00"]), ("\u0903\uAC01", &["\u0903", "\uAC01"]), ("\u0903\u0308\uAC01",
            &["\u0903\u0308", "\uAC01"]), ("\u0903\U0001F1E6", &["\u0903", "\U0001F1E6"]),
            ("\u0903\u0308\U0001F1E6", &["\u0903\u0308", "\U0001F1E6"]), ("\u0903\u0378",
            &["\u0903", "\u0378"]), ("\u0903\u0308\u0378", &["\u0903\u0308", "\u0378"]),
            ("\u1100\u0020", &["\u1100", "\u0020"]), ("\u1100\u0308\u0020", &["\u1100\u0308",
            "\u0020"]), ("\u1100\u000D", &["\u1100", "\u000D"]), ("\u1100\u0308\u000D",
            &["\u1100\u0308", "\u000D"]), ("\u1100\u000A", &["\u1100", "\u000A"]),
            ("\u1100\u0308\u000A", &["\u1100\u0308", "\u000A"]), ("\u1100\u0001", &["\u1100",
            "\u0001"]), ("\u1100\u0308\u0001", &["\u1100\u0308", "\u0001"]), ("\u1100\u0300",
            &["\u1100\u0300"]), ("\u1100\u0308\u0300", &["\u1100\u0308\u0300"]), ("\u1100\u1100",
            &["\u1100\u1100"]), ("\u1100\u0308\u1100", &["\u1100\u0308", "\u1100"]),
            ("\u1100\u1160", &["\u1100\u1160"]), ("\u1100\u0308\u1160", &["\u1100\u0308",
            "\u1160"]), ("\u1100\u11A8", &["\u1100", "\u11A8"]), ("\u1100\u0308\u11A8",
            &["\u1100\u0308", "\u11A8"]), ("\u1100\uAC00", &["\u1100\uAC00"]),
            ("\u1100\u0308\uAC00", &["\u1100\u0308", "\uAC00"]), ("\u1100\uAC01",
            &["\u1100\uAC01"]), ("\u1100\u0308\uAC01", &["\u1100\u0308", "\uAC01"]),
            ("\u1100\U0001F1E6", &["\u1100", "\U0001F1E6"]), ("\u1100\u0308\U0001F1E6",
            &["\u1100\u0308", "\U0001F1E6"]), ("\u1100\u0378", &["\u1100", "\u0378"]),
            ("\u1100\u0308\u0378", &["\u1100\u0308", "\u0378"]), ("\u1160\u0020", &["\u1160",
            "\u0020"]), ("\u1160\u0308\u0020", &["\u1160\u0308", "\u0020"]), ("\u1160\u000D",
            &["\u1160", "\u000D"]), ("\u1160\u0308\u000D", &["\u1160\u0308", "\u000D"]),
            ("\u1160\u000A", &["\u1160", "\u000A"]), ("\u1160\u0308\u000A", &["\u1160\u0308",
            "\u000A"]), ("\u1160\u0001", &["\u1160", "\u0001"]), ("\u1160\u0308\u0001",
            &["\u1160\u0308", "\u0001"]), ("\u1160\u0300", &["\u1160\u0300"]),
            ("\u1160\u0308\u0300", &["\u1160\u0308\u0300"]), ("\u1160\u1100", &["\u1160",
            "\u1100"]), ("\u1160\u0308\u1100", &["\u1160\u0308", "\u1100"]), ("\u1160\u1160",
            &["\u1160\u1160"]), ("\u1160\u0308\u1160", &["\u1160\u0308", "\u1160"]),
            ("\u1160\u11A8", &["\u1160\u11A8"]), ("\u1160\u0308\u11A8", &["\u1160\u0308",
            "\u11A8"]), ("\u1160\uAC00", &["\u1160", "\uAC00"]), ("\u1160\u0308\uAC00",
            &["\u1160\u0308", "\uAC00"]), ("\u1160\uAC01", &["\u1160", "\uAC01"]),
            ("\u1160\u0308\uAC01", &["\u1160\u0308", "\uAC01"]), ("\u1160\U0001F1E6", &["\u1160",
            "\U0001F1E6"]), ("\u1160\u0308\U0001F1E6", &["\u1160\u0308", "\U0001F1E6"]),
            ("\u1160\u0378", &["\u1160", "\u0378"]), ("\u1160\u0308\u0378", &["\u1160\u0308",
            "\u0378"]), ("\u11A8\u0020", &["\u11A8", "\u0020"]), ("\u11A8\u0308\u0020",
            &["\u11A8\u0308", "\u0020"]), ("\u11A8\u000D", &["\u11A8", "\u000D"]),
            ("\u11A8\u0308\u000D", &["\u11A8\u0308", "\u000D"]), ("\u11A8\u000A", &["\u11A8",
            "\u000A"]), ("\u11A8\u0308\u000A", &["\u11A8\u0308", "\u000A"]), ("\u11A8\u0001",
            &["\u11A8", "\u0001"]), ("\u11A8\u0308\u0001", &["\u11A8\u0308", "\u0001"]),
            ("\u11A8\u0300", &["\u11A8\u0300"]), ("\u11A8\u0308\u0300", &["\u11A8\u0308\u0300"]),
            ("\u11A8\u1100", &["\u11A8", "\u1100"]), ("\u11A8\u0308\u1100", &["\u11A8\u0308",
            "\u1100"]), ("\u11A8\u1160", &["\u11A8", "\u1160"]), ("\u11A8\u0308\u1160",
            &["\u11A8\u0308", "\u1160"]), ("\u11A8\u11A8", &["\u11A8\u11A8"]),
            ("\u11A8\u0308\u11A8", &["\u11A8\u0308", "\u11A8"]), ("\u11A8\uAC00", &["\u11A8",
            "\uAC00"]), ("\u11A8\u0308\uAC00", &["\u11A8\u0308", "\uAC00"]), ("\u11A8\uAC01",
            &["\u11A8", "\uAC01"]), ("\u11A8\u0308\uAC01", &["\u11A8\u0308", "\uAC01"]),
            ("\u11A8\U0001F1E6", &["\u11A8", "\U0001F1E6"]), ("\u11A8\u0308\U0001F1E6",
            &["\u11A8\u0308", "\U0001F1E6"]), ("\u11A8\u0378", &["\u11A8", "\u0378"]),
            ("\u11A8\u0308\u0378", &["\u11A8\u0308", "\u0378"]), ("\uAC00\u0020", &["\uAC00",
            "\u0020"]), ("\uAC00\u0308\u0020", &["\uAC00\u0308", "\u0020"]), ("\uAC00\u000D",
            &["\uAC00", "\u000D"]), ("\uAC00\u0308\u000D", &["\uAC00\u0308", "\u000D"]),
            ("\uAC00\u000A", &["\uAC00", "\u000A"]), ("\uAC00\u0308\u000A", &["\uAC00\u0308",
            "\u000A"]), ("\uAC00\u0001", &["\uAC00", "\u0001"]), ("\uAC00\u0308\u0001",
            &["\uAC00\u0308", "\u0001"]), ("\uAC00\u0300", &["\uAC00\u0300"]),
            ("\uAC00\u0308\u0300", &["\uAC00\u0308\u0300"]), ("\uAC00\u1100", &["\uAC00",
            "\u1100"]), ("\uAC00\u0308\u1100", &["\uAC00\u0308", "\u1100"]), ("\uAC00\u1160",
            &["\uAC00\u1160"]), ("\uAC00\u0308\u1160", &["\uAC00\u0308", "\u1160"]),
            ("\uAC00\u11A8", &["\uAC00\u11A8"]), ("\uAC00\u0308\u11A8", &["\uAC00\u0308",
            "\u11A8"]), ("\uAC00\uAC00", &["\uAC00", "\uAC00"]), ("\uAC00\u0308\uAC00",
            &["\uAC00\u0308", "\uAC00"]), ("\uAC00\uAC01", &["\uAC00", "\uAC01"]),
            ("\uAC00\u0308\uAC01", &["\uAC00\u0308", "\uAC01"]), ("\uAC00\U0001F1E6", &["\uAC00",
            "\U0001F1E6"]), ("\uAC00\u0308\U0001F1E6", &["\uAC00\u0308", "\U0001F1E6"]),
            ("\uAC00\u0378", &["\uAC00", "\u0378"]), ("\uAC00\u0308\u0378", &["\uAC00\u0308",
            "\u0378"]), ("\uAC01\u0020", &["\uAC01", "\u0020"]), ("\uAC01\u0308\u0020",
            &["\uAC01\u0308", "\u0020"]), ("\uAC01\u000D", &["\uAC01", "\u000D"]),
            ("\uAC01\u0308\u000D", &["\uAC01\u0308", "\u000D"]), ("\uAC01\u000A", &["\uAC01",
            "\u000A"]), ("\uAC01\u0308\u000A", &["\uAC01\u0308", "\u000A"]), ("\uAC01\u0001",
            &["\uAC01", "\u0001"]), ("\uAC01\u0308\u0001", &["\uAC01\u0308", "\u0001"]),
            ("\uAC01\u0300", &["\uAC01\u0300"]), ("\uAC01\u0308\u0300", &["\uAC01\u0308\u0300"]),
            ("\uAC01\u1100", &["\uAC01", "\u1100"]), ("\uAC01\u0308\u1100", &["\uAC01\u0308",
            "\u1100"]), ("\uAC01\u1160", &["\uAC01", "\u1160"]), ("\uAC01\u0308\u1160",
            &["\uAC01\u0308", "\u1160"]), ("\uAC01\u11A8", &["\uAC01\u11A8"]),
            ("\uAC01\u0308\u11A8", &["\uAC01\u0308", "\u11A8"]), ("\uAC01\uAC00", &["\uAC01",
            "\uAC00"]), ("\uAC01\u0308\uAC00", &["\uAC01\u0308", "\uAC00"]), ("\uAC01\uAC01",
            &["\uAC01", "\uAC01"]), ("\uAC01\u0308\uAC01", &["\uAC01\u0308", "\uAC01"]),
            ("\uAC01\U0001F1E6", &["\uAC01", "\U0001F1E6"]), ("\uAC01\u0308\U0001F1E6",
            &["\uAC01\u0308", "\U0001F1E6"]), ("\uAC01\u0378", &["\uAC01", "\u0378"]),
            ("\uAC01\u0308\u0378", &["\uAC01\u0308", "\u0378"]), ("\U0001F1E6\u0020",
            &["\U0001F1E6", "\u0020"]), ("\U0001F1E6\u0308\u0020", &["\U0001F1E6\u0308",
            "\u0020"]), ("\U0001F1E6\u000D", &["\U0001F1E6", "\u000D"]),
            ("\U0001F1E6\u0308\u000D", &["\U0001F1E6\u0308", "\u000D"]), ("\U0001F1E6\u000A",
            &["\U0001F1E6", "\u000A"]), ("\U0001F1E6\u0308\u000A", &["\U0001F1E6\u0308",
            "\u000A"]), ("\U0001F1E6\u0001", &["\U0001F1E6", "\u0001"]),
            ("\U0001F1E6\u0308\u0001", &["\U0001F1E6\u0308", "\u0001"]), ("\U0001F1E6\u0300",
            &["\U0001F1E6\u0300"]), ("\U0001F1E6\u0308\u0300", &["\U0001F1E6\u0308\u0300"]),
            ("\U0001F1E6\u1100", &["\U0001F1E6", "\u1100"]), ("\U0001F1E6\u0308\u1100",
            &["\U0001F1E6\u0308", "\u1100"]), ("\U0001F1E6\u1160", &["\U0001F1E6", "\u1160"]),
            ("\U0001F1E6\u0308\u1160", &["\U0001F1E6\u0308", "\u1160"]), ("\U0001F1E6\u11A8",
            &["\U0001F1E6", "\u11A8"]), ("\U0001F1E6\u0308\u11A8", &["\U0001F1E6\u0308",
            "\u11A8"]), ("\U0001F1E6\uAC00", &["\U0001F1E6", "\uAC00"]),
            ("\U0001F1E6\u0308\uAC00", &["\U0001F1E6\u0308", "\uAC00"]), ("\U0001F1E6\uAC01",
            &["\U0001F1E6", "\uAC01"]), ("\U0001F1E6\u0308\uAC01", &["\U0001F1E6\u0308",
            "\uAC01"]), ("\U0001F1E6\U0001F1E6", &["\U0001F1E6\U0001F1E6"]),
            ("\U0001F1E6\u0308\U0001F1E6", &["\U0001F1E6\u0308", "\U0001F1E6"]),
            ("\U0001F1E6\u0378", &["\U0001F1E6", "\u0378"]), ("\U0001F1E6\u0308\u0378",
            &["\U0001F1E6\u0308", "\u0378"]), ("\u0378\u0020", &["\u0378", "\u0020"]),
            ("\u0378\u0308\u0020", &["\u0378\u0308", "\u0020"]), ("\u0378\u000D", &["\u0378",
            "\u000D"]), ("\u0378\u0308\u000D", &["\u0378\u0308", "\u000D"]), ("\u0378\u000A",
            &["\u0378", "\u000A"]), ("\u0378\u0308\u000A", &["\u0378\u0308", "\u000A"]),
            ("\u0378\u0001", &["\u0378", "\u0001"]), ("\u0378\u0308\u0001", &["\u0378\u0308",
            "\u0001"]), ("\u0378\u0300", &["\u0378\u0300"]), ("\u0378\u0308\u0300",
            &["\u0378\u0308\u0300"]), ("\u0378\u1100", &["\u0378", "\u1100"]),
            ("\u0378\u0308\u1100", &["\u0378\u0308", "\u1100"]), ("\u0378\u1160", &["\u0378",
            "\u1160"]), ("\u0378\u0308\u1160", &["\u0378\u0308", "\u1160"]), ("\u0378\u11A8",
            &["\u0378", "\u11A8"]), ("\u0378\u0308\u11A8", &["\u0378\u0308", "\u11A8"]),
            ("\u0378\uAC00", &["\u0378", "\uAC00"]), ("\u0378\u0308\uAC00", &["\u0378\u0308",
            "\uAC00"]), ("\u0378\uAC01", &["\u0378", "\uAC01"]), ("\u0378\u0308\uAC01",
            &["\u0378\u0308", "\uAC01"]), ("\u0378\U0001F1E6", &["\u0378", "\U0001F1E6"]),
            ("\u0378\u0308\U0001F1E6", &["\u0378\u0308", "\U0001F1E6"]), ("\u0378\u0378",
            &["\u0378", "\u0378"]), ("\u0378\u0308\u0378", &["\u0378\u0308", "\u0378"]),
            ("\u0061\U0001F1E6\u0062", &["\u0061", "\U0001F1E6", "\u0062"]),
            ("\U0001F1F7\U0001F1FA", &["\U0001F1F7\U0001F1FA"]),
            ("\U0001F1F7\U0001F1FA\U0001F1F8", &["\U0001F1F7\U0001F1FA\U0001F1F8"]),
            ("\U0001F1F7\U0001F1FA\U0001F1F8\U0001F1EA",
            &["\U0001F1F7\U0001F1FA\U0001F1F8\U0001F1EA"]),
            ("\U0001F1F7\U0001F1FA\u200B\U0001F1F8\U0001F1EA", &["\U0001F1F7\U0001F1FA", "\u200B",
            "\U0001F1F8\U0001F1EA"]), ("\U0001F1E6\U0001F1E7\U0001F1E8",
            &["\U0001F1E6\U0001F1E7\U0001F1E8"]), ("\U0001F1E6\u200D\U0001F1E7\U0001F1E8",
            &["\U0001F1E6\u200D", "\U0001F1E7\U0001F1E8"]),
            ("\U0001F1E6\U0001F1E7\u200D\U0001F1E8", &["\U0001F1E6\U0001F1E7\u200D",
            "\U0001F1E8"]), ("\u0020\u200D\u0646", &["\u0020\u200D", "\u0646"]),
            ("\u0646\u200D\u0020", &["\u0646\u200D", "\u0020"]),
        ];

        let test_diff = [
            ("\u0020\u0903", &["\u0020\u0903"], &["\u0020", "\u0903"]), ("\u0020\u0308\u0903",
            &["\u0020\u0308\u0903"], &["\u0020\u0308", "\u0903"]), ("\u000D\u0308\u0903",
            &["\u000D", "\u0308\u0903"], &["\u000D", "\u0308", "\u0903"]), ("\u000A\u0308\u0903",
            &["\u000A", "\u0308\u0903"], &["\u000A", "\u0308", "\u0903"]), ("\u0001\u0308\u0903",
            &["\u0001", "\u0308\u0903"], &["\u0001", "\u0308", "\u0903"]), ("\u0300\u0903",
            &["\u0300\u0903"], &["\u0300", "\u0903"]), ("\u0300\u0308\u0903",
            &["\u0300\u0308\u0903"], &["\u0300\u0308", "\u0903"]), ("\u0903\u0903",
            &["\u0903\u0903"], &["\u0903", "\u0903"]), ("\u0903\u0308\u0903",
            &["\u0903\u0308\u0903"], &["\u0903\u0308", "\u0903"]), ("\u1100\u0903",
            &["\u1100\u0903"], &["\u1100", "\u0903"]), ("\u1100\u0308\u0903",
            &["\u1100\u0308\u0903"], &["\u1100\u0308", "\u0903"]), ("\u1160\u0903",
            &["\u1160\u0903"], &["\u1160", "\u0903"]), ("\u1160\u0308\u0903",
            &["\u1160\u0308\u0903"], &["\u1160\u0308", "\u0903"]), ("\u11A8\u0903",
            &["\u11A8\u0903"], &["\u11A8", "\u0903"]), ("\u11A8\u0308\u0903",
            &["\u11A8\u0308\u0903"], &["\u11A8\u0308", "\u0903"]), ("\uAC00\u0903",
            &["\uAC00\u0903"], &["\uAC00", "\u0903"]), ("\uAC00\u0308\u0903",
            &["\uAC00\u0308\u0903"], &["\uAC00\u0308", "\u0903"]), ("\uAC01\u0903",
            &["\uAC01\u0903"], &["\uAC01", "\u0903"]), ("\uAC01\u0308\u0903",
            &["\uAC01\u0308\u0903"], &["\uAC01\u0308", "\u0903"]), ("\U0001F1E6\u0903",
            &["\U0001F1E6\u0903"], &["\U0001F1E6", "\u0903"]), ("\U0001F1E6\u0308\u0903",
            &["\U0001F1E6\u0308\u0903"], &["\U0001F1E6\u0308", "\u0903"]), ("\u0378\u0903",
            &["\u0378\u0903"], &["\u0378", "\u0903"]), ("\u0378\u0308\u0903",
            &["\u0378\u0308\u0903"], &["\u0378\u0308", "\u0903"]),
        ];

        for &(s, g) in test_same.iter() {
            // test forward iterator
            assert!(order::equals(s.graphemes(true), g.iter().map(|&x| x)));
            assert!(order::equals(s.graphemes(false), g.iter().map(|&x| x)));

            // test reverse iterator
            assert!(order::equals(s.graphemes(true).rev(), g.iter().rev().map(|&x| x)));
            assert!(order::equals(s.graphemes(false).rev(), g.iter().rev().map(|&x| x)));
        }

        for &(s, gt, gf) in test_diff.iter() {
            // test forward iterator
            assert!(order::equals(s.graphemes(true), gt.iter().map(|&x| x)));
            assert!(order::equals(s.graphemes(false), gf.iter().map(|&x| x)));

            // test reverse iterator
            assert!(order::equals(s.graphemes(true).rev(), gt.iter().rev().map(|&x| x)));
            assert!(order::equals(s.graphemes(false).rev(), gf.iter().rev().map(|&x| x)));
        }

        // test the indices iterators
        let s = "a̐éö̲\r\n";
        let gr_inds = s.grapheme_indices(true).collect::<Vec<(uint, &str)>>();
        assert_eq!(gr_inds.as_slice(), &[(0u, "a̐"), (3, "é"), (6, "ö̲"), (11, "\r\n")]);
        let gr_inds = s.grapheme_indices(true).rev().collect::<Vec<(uint, &str)>>();
        assert_eq!(gr_inds.as_slice(), &[(11, "\r\n"), (6, "ö̲"), (3, "é"), (0u, "a̐")]);
        let mut gr_inds = s.grapheme_indices(true);
        let e1 = gr_inds.size_hint();
        assert_eq!(e1, (1, Some(13)));
        let c = gr_inds.count();
        assert_eq!(c, 4);
        let e2 = gr_inds.size_hint();
        assert_eq!(e2, (0, Some(0)));

        // make sure the reverse iterator does the right thing with "\n" at beginning of string
        let s = "\n\r\n\r";
        let gr = s.graphemes(true).rev().collect::<Vec<&str>>();
        assert_eq!(gr.as_slice(), &["\r", "\r\n", "\n"]);
    }

    #[test]
    fn test_split_strator() {
        fn t(s: &str, sep: &str, u: &[&str]) {
            let v: Vec<&str> = s.split_str(sep).collect();
            assert_eq!(v.as_slice(), u.as_slice());
        }
        t("--1233345--", "12345", ["--1233345--"]);
        t("abc::hello::there", "::", ["abc", "hello", "there"]);
        t("::hello::there", "::", ["", "hello", "there"]);
        t("hello::there::", "::", ["hello", "there", ""]);
        t("::hello::there::", "::", ["", "hello", "there", ""]);
        t("ประเทศไทย中华Việt Nam", "中华", ["ประเทศไทย", "Việt Nam"]);
        t("zzXXXzzYYYzz", "zz", ["", "XXX", "YYY", ""]);
        t("zzXXXzYYYz", "XXX", ["zz", "zYYYz"]);
        t(".XXX.YYY.", ".", ["", "XXX", "YYY", ""]);
        t("", ".", [""]);
        t("zz", "zz", ["",""]);
        t("ok", "z", ["ok"]);
        t("zzz", "zz", ["","z"]);
        t("zzzzz", "zz", ["","","z"]);
    }

    #[test]
    fn test_str_default() {
        use std::default::Default;
        fn t<S: Default + Str>() {
            let s: S = Default::default();
            assert_eq!(s.as_slice(), "");
        }

        t::<&str>();
        t::<String>();
    }

    #[test]
    fn test_str_container() {
        fn sum_len<S: Collection>(v: &[S]) -> uint {
            v.iter().map(|x| x.len()).sum()
        }

        let s = String::from_str("01234");
        assert_eq!(5, sum_len(["012", "", "34"]));
        assert_eq!(5, sum_len([String::from_str("01"), String::from_str("2"),
                               String::from_str("34"), String::from_str("")]));
        assert_eq!(5, sum_len([s.as_slice()]));
    }

    #[test]
    fn test_str_from_utf8() {
        let xs = b"hello";
        assert_eq!(from_utf8(xs), Some("hello"));

        let xs = "ศไทย中华Việt Nam".as_bytes();
        assert_eq!(from_utf8(xs), Some("ศไทย中华Việt Nam"));

        let xs = b"hello\xFF";
        assert_eq!(from_utf8(xs), None);
    }

    #[test]
    fn test_maybe_owned_traits() {
        let s = Slice("abcde");
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_slice(), "abcde");
        assert_eq!(String::from_str(s.as_slice()).as_slice(), "abcde");
        assert_eq!(format!("{}", s).as_slice(), "abcde");
        assert!(s.lt(&Owned(String::from_str("bcdef"))));
        assert_eq!(Slice(""), Default::default());

        let o = Owned(String::from_str("abcde"));
        assert_eq!(o.len(), 5);
        assert_eq!(o.as_slice(), "abcde");
        assert_eq!(String::from_str(o.as_slice()).as_slice(), "abcde");
        assert_eq!(format!("{}", o).as_slice(), "abcde");
        assert!(o.lt(&Slice("bcdef")));
        assert_eq!(Owned(String::from_str("")), Default::default());

        assert!(s.cmp(&o) == Equal);
        assert!(s.equiv(&o));

        assert!(o.cmp(&s) == Equal);
        assert!(o.equiv(&s));
    }

    #[test]
    fn test_maybe_owned_methods() {
        let s = Slice("abcde");
        assert!(s.is_slice());
        assert!(!s.is_owned());

        let o = Owned(String::from_str("abcde"));
        assert!(!o.is_slice());
        assert!(o.is_owned());
    }

    #[test]
    fn test_maybe_owned_clone() {
        assert_eq!(Owned(String::from_str("abcde")), Slice("abcde").clone());
        assert_eq!(Owned(String::from_str("abcde")), Owned(String::from_str("abcde")).clone());
        assert_eq!(Slice("abcde"), Slice("abcde").clone());
        assert_eq!(Slice("abcde"), Owned(String::from_str("abcde")).clone());
    }

    #[test]
    fn test_maybe_owned_into_string() {
        assert_eq!(Slice("abcde").into_string(), String::from_str("abcde"));
        assert_eq!(Owned(String::from_str("abcde")).into_string(),
                   String::from_str("abcde"));
    }

    #[test]
    fn test_into_maybe_owned() {
        assert_eq!("abcde".into_maybe_owned(), Slice("abcde"));
        assert_eq!((String::from_str("abcde")).into_maybe_owned(), Slice("abcde"));
        assert_eq!("abcde".into_maybe_owned(), Owned(String::from_str("abcde")));
        assert_eq!((String::from_str("abcde")).into_maybe_owned(),
                   Owned(String::from_str("abcde")));
    }
}

#[cfg(test)]
mod bench {
    use test::Bencher;
    use test::black_box;
    use super::*;
    use std::option::{None, Some};
    use std::iter::{Iterator, DoubleEndedIterator};
    use std::collections::Collection;

    #[bench]
    fn char_iterator(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| s.chars().count());
    }

    #[bench]
    fn char_iterator_for(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";

        b.iter(|| {
            for ch in s.chars() { black_box(ch) }
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
            for ch in s.chars().rev() { black_box(ch) }
        });
    }

    #[bench]
    fn char_indicesator(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.char_len();

        b.iter(|| assert_eq!(s.char_indices().count(), len));
    }

    #[bench]
    fn char_indicesator_rev(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.char_len();

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

        b.iter(|| assert_eq!(s.split(&[' ']).count(), len));
    }

    #[bench]
    fn is_utf8_100_ascii(b: &mut Bencher) {

        let s = b"Hello there, the quick brown fox jumped over the lazy dog! \
                  Lorem ipsum dolor sit amet, consectetur. ";

        assert_eq!(100, s.len());
        b.iter(|| {
            is_utf8(s)
        });
    }

    #[bench]
    fn is_utf8_100_multibyte(b: &mut Bencher) {
        let s = "𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰".as_bytes();
        assert_eq!(100, s.len());
        b.iter(|| {
            is_utf8(s)
        });
    }

    #[bench]
    fn bench_connect(b: &mut Bencher) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let sep = "→";
        let v = [s, s, s, s, s, s, s, s, s, s];
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
