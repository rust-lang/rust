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

Unicode string manipulation (`str` type)

# Basic Usage

Rust's string type is one of the core primitive types of the language. While
represented by the name `str`, the name `str` is not actually a valid type in
Rust. Each string must also be decorated with its ownership. This means that
there are three common kinds of strings in rust:

* `~str` - This is an owned string. This type obeys all of the normal semantics
           of the `~T` types, meaning that it has one, and only one, owner. This
           type cannot be implicitly copied, and is moved out of when passed to
           other functions.

* `@str` - This is a managed string. Similarly to `@T`, this type can be
           implicitly copied, and each implicit copy will increment the
           reference count to the string. This means that there is no "true
           owner" of the string, and the string will be deallocated when the
           reference count reaches 0.

* `&str` - Finally, this is the borrowed string type. This type of string can
           only be created from one of the other two kinds of strings. As the
           name "borrowed" implies, this type of string is owned elsewhere, and
           this string cannot be moved out of.

As an example, here's a few different kinds of strings.

```rust
#[feature(managed_boxes)];

fn main() {
    let owned_string = ~"I am an owned string";
    let managed_string = @"This string is garbage-collected";
    let borrowed_string1 = "This string is borrowed with the 'static lifetime";
    let borrowed_string2: &str = owned_string;   // owned strings can be borrowed
    let borrowed_string3: &str = managed_string; // managed strings can also be borrowed
}
 ```

From the example above, you can see that rust has 3 different kinds of string
literals. The owned/managed literals correspond to the owned/managed string
types, but the "borrowed literal" is actually more akin to C's concept of a
static string.

When a string is declared without a `~` or `@` sigil, then the string is
allocated statically in the rodata of the executable/library. The string then
has the type `&'static str` meaning that the string is valid for the `'static`
lifetime, otherwise known as the lifetime of the entire program. As can be
inferred from the type, these static strings are not mutable.

# Mutability

Many languages have immutable strings by default, and rust has a particular
flavor on this idea. As with the rest of Rust types, strings are immutable by
default. If a string is declared as `mut`, however, it may be mutated. This
works the same way as the rest of Rust's type system in the sense that if
there's a mutable reference to a string, there may only be one mutable reference
to that string. With these guarantees, strings can easily transition between
being mutable/immutable with the same benefits of having mutable strings in
other languages.

```rust
let mut buf = ~"testing";
buf.push_char(' ');
buf.push_str("123");
assert_eq!(buf, ~"testing 123");
 ```

# Representation

Rust's string type, `str`, is a sequence of unicode codepoints encoded as a
stream of UTF-8 bytes. All safely-created strings are guaranteed to be validly
encoded UTF-8 sequences. Additionally, strings are not null-terminated
and can contain null codepoints.

The actual representation of strings have direct mappings to vectors:

* `~str` is the same as `~[u8]`
* `&str` is the same as `&[u8]`
* `@str` is the same as `@[u8]`

*/

use at_vec;
use cast;
use cast::transmute;
use char;
use char::Char;
use clone::{Clone, DeepClone};
use container::{Container, Mutable};
use iter::{Iterator, FromIterator, Extendable, range};
use iter::{Filter, AdditiveIterator, Map};
use iter::{Invert, DoubleEndedIterator, ExactSize};
use libc;
use num::{Saturating};
use option::{None, Option, Some};
use ptr;
use ptr::RawPtr;
use to_str::ToStr;
use from_str::FromStr;
use uint;
use vec;
use vec::{OwnedVector, OwnedCopyableVector, ImmutableVector, MutableVector};
use default::Default;
use send_str::{SendStr, SendStrOwned};
use unstable::raw::Repr;

/*
Section: Conditions
*/

condition! {
    pub not_utf8: (~str) -> ~str;
}

/*
Section: Creating a string
*/

/// Consumes a vector of bytes to create a new utf-8 string
///
/// # Failure
///
/// Raises the `not_utf8` condition if invalid UTF-8
pub fn from_utf8_owned(vv: ~[u8]) -> ~str {
    use str::not_utf8::cond;

    if !is_utf8(vv) {
        let first_bad_byte = *vv.iter().find(|&b| !is_utf8([*b])).unwrap();
        cond.raise(format!("from_utf8: input is not UTF-8; first bad byte is {}",
                           first_bad_byte))
    } else {
        unsafe { raw::from_utf8_owned(vv) }
    }
}

/// Consumes a vector of bytes to create a new utf-8 string.
/// Returns None if the vector contains invalid UTF-8.
pub fn from_utf8_owned_opt(vv: ~[u8]) -> Option<~str> {
    if is_utf8(vv) {
        Some(unsafe { raw::from_utf8_owned(vv) })
    } else {
        None
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
pub fn from_utf8<'a>(v: &'a [u8]) -> &'a str {
    from_utf8_opt(v).expect("from_utf8: not utf-8")
}

/// Converts a vector to a string slice without performing any allocations.
///
/// Returns None if the slice is not utf-8.
pub fn from_utf8_opt<'a>(v: &'a [u8]) -> Option<&'a str> {
    if is_utf8(v) {
        Some(unsafe { raw::from_utf8(v) })
    } else { None }
}

impl ToStr for ~str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
}

impl FromStr for ~str {
    #[inline]
    fn from_str(s: &str) -> Option<~str> { Some(s.to_owned()) }
}

impl<'a> ToStr for &'a str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
}

impl ToStr for @str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
}

impl<'a> FromStr for @str {
    #[inline]
    fn from_str(s: &str) -> Option<@str> { Some(s.to_managed()) }
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

/// Methods for vectors of strings
pub trait StrVector {
    /// Concatenate a vector of strings.
    fn concat(&self) -> ~str;

    /// Concatenate a vector of strings, placing a given separator between each.
    fn connect(&self, sep: &str) -> ~str;
}

impl<'a, S: Str> StrVector for &'a [S] {
    fn concat(&self) -> ~str {
        if self.is_empty() { return ~""; }

        // `len` calculation may overflow but push_str but will check boundaries
        let len = self.iter().map(|s| s.as_slice().len()).sum();

        let mut result = with_capacity(len);

        for s in self.iter() {
            result.push_str(s.as_slice())
        }
        result
    }

    fn connect(&self, sep: &str) -> ~str {
        if self.is_empty() { return ~""; }

        // concat is faster
        if sep.is_empty() { return self.concat(); }

        // this is wrong without the guarantee that `self` is non-empty
        // `len` calculation may overflow but push_str but will check boundaries
        let len = sep.len() * (self.len() - 1)
            + self.iter().map(|s| s.as_slice().len()).sum();
        let mut result = with_capacity(len);
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

impl<'a> CharEq for 'a |char| -> bool {
    #[inline]
    fn matches(&self, c: char) -> bool { (*self)(c) }

    fn only_ascii(&self) -> bool { false }
}

impl CharEq for extern "Rust" fn(char) -> bool {
    #[inline]
    fn matches(&self, c: char) -> bool { (*self)(c) }

    fn only_ascii(&self) -> bool { false }
}

impl<'a, C: CharEq> CharEq for &'a [C] {
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

/// External iterator for a string's characters.
/// Use with the `std::iter` module.
#[deriving(Clone)]
pub struct CharIterator<'a> {
    /// The slice remaining to be iterated
    priv string: &'a str,
}

impl<'a> Iterator<char> for CharIterator<'a> {
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

impl<'a> DoubleEndedIterator<char> for CharIterator<'a> {
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
pub struct CharOffsetIterator<'a> {
    /// The original string to be iterated
    priv string: &'a str,
    priv iter: CharIterator<'a>,
}

impl<'a> Iterator<(uint, char)> for CharOffsetIterator<'a> {
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

impl<'a> DoubleEndedIterator<(uint, char)> for CharOffsetIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, char)> {
        self.iter.next_back().map(|ch| {
            let offset = self.iter.string.len() +
                    self.iter.string.as_ptr() as uint - self.string.as_ptr() as uint;
            (offset, ch)
        })
    }
}

/// External iterator for a string's characters in reverse order.
/// Use with the `std::iter` module.
pub type CharRevIterator<'a> = Invert<CharIterator<'a>>;

/// External iterator for a string's characters and their byte offsets in reverse order.
/// Use with the `std::iter` module.
pub type CharOffsetRevIterator<'a> = Invert<CharOffsetIterator<'a>>;

/// External iterator for a string's bytes.
/// Use with the `std::iter` module.
pub type ByteIterator<'a> =
    Map<'a, &'a u8, u8, vec::VecIterator<'a, u8>>;

/// External iterator for a string's bytes in reverse order.
/// Use with the `std::iter` module.
pub type ByteRevIterator<'a> = Invert<ByteIterator<'a>>;

/// An iterator over the substrings of a string, separated by `sep`.
#[deriving(Clone)]
pub struct CharSplitIterator<'a, Sep> {
    /// The slice remaining to be iterated
    priv string: &'a str,
    priv sep: Sep,
    /// Whether an empty string at the end is allowed
    priv allow_trailing_empty: bool,
    priv only_ascii: bool,
    priv finished: bool,
}

/// An iterator over the substrings of a string, separated by `sep`,
/// starting from the back of the string.
pub type CharRSplitIterator<'a, Sep> = Invert<CharSplitIterator<'a, Sep>>;

/// An iterator over the substrings of a string, separated by `sep`,
/// splitting at most `count` times.
#[deriving(Clone)]
pub struct CharSplitNIterator<'a, Sep> {
    priv iter: CharSplitIterator<'a, Sep>,
    /// The number of splits remaining
    priv count: uint,
    priv invert: bool,
}

/// An iterator over the words of a string, separated by an sequence of whitespace
pub type WordIterator<'a> =
    Filter<'a, &'a str, CharSplitIterator<'a, extern "Rust" fn(char) -> bool>>;

/// An iterator over the lines of a string, separated by either `\n` or (`\r\n`).
pub type AnyLineIterator<'a> =
    Map<'a, &'a str, &'a str, CharSplitIterator<'a, char>>;

impl<'a, Sep> CharSplitIterator<'a, Sep> {
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

impl<'a, Sep: CharEq> Iterator<&'a str> for CharSplitIterator<'a, Sep> {
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
for CharSplitIterator<'a, Sep> {
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
            for (idx, byte) in self.string.bytes().enumerate().invert() {
                if self.sep.matches(byte as char) && byte < 128u8 {
                    next_split = Some((idx, idx + 1));
                    break;
                }
            }
        } else {
            for (idx, ch) in self.string.char_indices_rev() {
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

impl<'a, Sep: CharEq> Iterator<&'a str> for CharSplitNIterator<'a, Sep> {
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

/// An iterator over the start and end indices of the matches of a
/// substring within a larger string
#[deriving(Clone)]
pub struct MatchesIndexIterator<'a> {
    priv haystack: &'a str,
    priv needle: &'a str,
    priv position: uint,
}

/// An iterator over the substrings of a string separated by a given
/// search string
#[deriving(Clone)]
pub struct StrSplitIterator<'a> {
    priv it: MatchesIndexIterator<'a>,
    priv last_end: uint,
    priv finished: bool
}

impl<'a> Iterator<(uint, uint)> for MatchesIndexIterator<'a> {
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

impl<'a> Iterator<&'a str> for StrSplitIterator<'a> {
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

// Helper functions used for Unicode normalization
fn canonical_sort(comb: &mut [(char, u8)]) {
    use iter::range;
    use tuple::CopyableTuple;

    let len = comb.len();
    for i in range(0, len) {
        let mut swapped = false;
        for j in range(1, len-i) {
            let classA = comb[j-1].second();
            let classB = comb[j].second();
            if classA != 0 && classB != 0 && classA > classB {
                comb.swap(j-1, j);
                swapped = true;
            }
        }
        if !swapped { break; }
    }
}

#[deriving(Clone)]
enum NormalizationForm {
    NFD,
    NFKD
}

/// External iterator for a string's normalization's characters.
/// Use with the `std::iter` module.
#[deriving(Clone)]
struct NormalizationIterator<'a> {
    priv kind: NormalizationForm,
    priv iter: CharIterator<'a>,
    priv buffer: ~[(char, u8)],
    priv sorted: bool
}

impl<'a> Iterator<char> for NormalizationIterator<'a> {
    #[inline]
    fn next(&mut self) -> Option<char> {
        use unicode::decompose::canonical_combining_class;

        match self.buffer.head_opt() {
            Some(&(c, 0)) => {
                self.sorted = false;
                self.buffer.shift();
                return Some(c);
            }
            Some(&(c, _)) if self.sorted => {
                self.buffer.shift();
                return Some(c);
            }
            _ => self.sorted = false
        }

        let decomposer = match self.kind {
            NFD => char::decompose_canonical,
            NFKD => char::decompose_compatible
        };

        if !self.sorted {
            for ch in self.iter {
                decomposer(ch, |d| {
                    let class = canonical_combining_class(d);
                    if class == 0 && !self.sorted {
                        canonical_sort(self.buffer);
                        self.sorted = true;
                    }
                    self.buffer.push((d, class));
                });
                if self.sorted { break }
            }
        }

        if !self.sorted {
            canonical_sort(self.buffer);
            self.sorted = true;
        }

        match self.buffer.shift_opt() {
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
    for (start, end) in s.match_indices(from) {
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

// share the implementation of the lang-item vs. non-lang-item
// eq_slice.
#[inline]
fn eq_slice_(a: &str, b: &str) -> bool {
    a.len() == b.len() && unsafe {
        libc::memcmp(a.as_ptr() as *libc::c_void,
                     b.as_ptr() as *libc::c_void,
                     a.len() as libc::size_t) == 0
    }
}

/// Bytewise slice equality
#[cfg(not(test))]
#[lang="str_eq"]
#[inline]
pub fn eq_slice(a: &str, b: &str) -> bool {
    eq_slice_(a, b)
}

/// Bytewise slice equality
#[cfg(test)]
#[inline]
pub fn eq_slice(a: &str, b: &str) -> bool {
    eq_slice_(a, b)
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
            // UTF8-tail   = %x80-BF
            match w {
                2 => if unsafe_get(v, i + 1) & 192u8 != TAG_CONT_U8 {
                    return false
                },
                3 => match (v_i,
                            unsafe_get(v, i + 1),
                            unsafe_get(v, i + 2) & 192u8) {
                    (0xE0        , 0xA0 .. 0xBF, TAG_CONT_U8) => (),
                    (0xE1 .. 0xEC, 0x80 .. 0xBF, TAG_CONT_U8) => (),
                    (0xED        , 0x80 .. 0x9F, TAG_CONT_U8) => (),
                    (0xEE .. 0xEF, 0x80 .. 0xBF, TAG_CONT_U8) => (),
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
pub fn utf16_chars(v: &[u16], f: |char|) {
    let len = v.len();
    let mut i = 0u;
    while (i < len && v[i] != 0u16) {
        let u = v[i];

        if  u <= 0xD7FF_u16 || u >= 0xE000_u16 {
            f(unsafe { cast::transmute(u as u32) });
            i += 1u;

        } else {
            let u2 = v[i+1u];
            assert!(u >= 0xD800_u16 && u <= 0xDBFF_u16);
            assert!(u2 >= 0xDC00_u16 && u2 <= 0xDFFF_u16);
            let mut c: u32 = (u - 0xD800_u16) as u32;
            c = c << 10;
            c |= (u2 - 0xDC00_u16) as u32;
            c |= 0x1_0000_u32;
            f(unsafe { cast::transmute(c) });
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

/// Struct that contains a `char` and the index of the first byte of
/// the next `char` in a string.  This can be used as a data structure
/// for iterating over the UTF-8 bytes of a string.
pub struct CharRange {
    /// Current `char`
    ch: char,
    /// Index of the first byte of the next `char`
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

/// Unsafe operations
pub mod raw {
    use cast;
    use container::Container;
    use libc;
    use ptr;
    use ptr::RawPtr;
    use str::{is_utf8, OwnedStr, StrSlice};
    use vec;
    use vec::{MutableVector, ImmutableVector, OwnedVector};
    use unstable::raw::Slice;

    /// Create a Rust string from a *u8 buffer of the given length
    pub unsafe fn from_buf_len(buf: *u8, len: uint) -> ~str {
        let mut v: ~[u8] = vec::with_capacity(len);
        ptr::copy_memory(v.as_mut_ptr(), buf, len);
        v.set_len(len);

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

    /// Converts a slice of bytes to a string slice without checking
    /// that the string contains valid UTF-8.
    pub unsafe fn from_utf8<'a>(v: &'a [u8]) -> &'a str {
        cast::transmute(v)
    }

    /// Converts an owned vector of bytes to a new owned string. This assumes
    /// that the utf-8-ness of the vector has already been validated
    #[inline]
    pub unsafe fn from_utf8_owned(v: ~[u8]) -> ~str {
        cast::transmute(v)
    }

    /// Converts a byte to a string.
    pub unsafe fn from_byte(u: u8) -> ~str { from_utf8_owned(~[u]) }

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
        cast::transmute(Slice {
                data: s.as_ptr().offset(begin as int),
                len: end - begin,
            })
    }

    /// Appends a byte to a string.
    /// The caller must preserve the valid UTF-8 property.
    #[inline]
    pub unsafe fn push_byte(s: &mut ~str, b: u8) {
        as_owned_vec(s).push(b)
    }

    /// Appends a vector of bytes to a string.
    /// The caller must preserve the valid UTF-8 property.
    #[inline]
    pub unsafe fn push_bytes(s: &mut ~str, bytes: &[u8]) {
        vec::bytes::push_bytes(as_owned_vec(s), bytes);
    }

    /// Removes the last byte from a string and returns it.
    /// The caller must preserve the valid UTF-8 property.
    pub unsafe fn pop_byte(s: &mut ~str) -> u8 {
        let len = s.len();
        assert!((len > 0u));
        let b = s[len - 1u];
        s.set_len(len - 1);
        return b;
    }

    /// Removes the first byte from a string and returns it.
    /// The caller must preserve the valid UTF-8 property.
    pub unsafe fn shift_byte(s: &mut ~str) -> u8 {
        let len = s.len();
        assert!((len > 0u));
        let b = s[0];
        *s = s.slice(1, len).to_owned();
        return b;
    }

    /// Access the str in its vector representation.
    /// The caller must preserve the valid UTF-8 property when modifying.
    #[inline]
    pub unsafe fn as_owned_vec<'a>(s: &'a mut ~str) -> &'a mut ~[u8] {
        cast::transmute(s)
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
            let b = a.as_ptr();
            let c = from_buf_len(b, 3u);
            assert_eq!(c, ~"AAA");
        }
    }
}

/*
Section: Trait implementations
*/

#[cfg(not(test))]
#[allow(missing_doc)]
pub mod traits {
    use container::Container;
    use cmp::{TotalOrd, Ordering, Less, Equal, Greater, Eq, Ord, Equiv, TotalEq};
    use iter::Iterator;
    use ops::Add;
    use option::{Some, None};
    use str::{Str, StrSlice, OwnedStr, eq_slice};

    impl<'a> Add<&'a str,~str> for &'a str {
        #[inline]
        fn add(&self, rhs: & &'a str) -> ~str {
            let mut ret = self.to_owned();
            ret.push_str(*rhs);
            ret
        }
    }

    impl<'a> TotalOrd for &'a str {
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

    impl TotalOrd for ~str {
        #[inline]
        fn cmp(&self, other: &~str) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl TotalOrd for @str {
        #[inline]
        fn cmp(&self, other: &@str) -> Ordering { self.as_slice().cmp(&other.as_slice()) }
    }

    impl<'a> Eq for &'a str {
        #[inline]
        fn eq(&self, other: & &'a str) -> bool {
            eq_slice((*self), (*other))
        }
        #[inline]
        fn ne(&self, other: & &'a str) -> bool { !(*self).eq(other) }
    }

    impl Eq for ~str {
        #[inline]
        fn eq(&self, other: &~str) -> bool {
            eq_slice((*self), (*other))
        }
    }

    impl Eq for @str {
        #[inline]
        fn eq(&self, other: &@str) -> bool {
            eq_slice((*self), (*other))
        }
    }

    impl<'a> TotalEq for &'a str {
        #[inline]
        fn equals(&self, other: & &'a str) -> bool {
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

    impl<'a> Ord for &'a str {
        #[inline]
        fn lt(&self, other: & &'a str) -> bool { self.cmp(other) == Less }
    }

    impl Ord for ~str {
        #[inline]
        fn lt(&self, other: &~str) -> bool { self.cmp(other) == Less }
    }

    impl Ord for @str {
        #[inline]
        fn lt(&self, other: &@str) -> bool { self.cmp(other) == Less }
    }

    impl<'a, S: Str> Equiv<S> for &'a str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(*self, other.as_slice()) }
    }

    impl<'a, S: Str> Equiv<S> for @str {
        #[inline]
        fn equiv(&self, other: &S) -> bool { eq_slice(*self, other.as_slice()) }
    }

    impl<'a, S: Str> Equiv<S> for ~str {
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

impl<'a> Str for &'a str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str { *self }

    #[inline]
    fn into_owned(self) -> ~str { self.to_owned() }
}

impl<'a> Str for ~str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str {
        let s: &'a str = *self; s
    }

    #[inline]
    fn into_owned(self) -> ~str { self }
}

impl<'a> Str for @str {
    #[inline]
    fn as_slice<'a>(&'a self) -> &'a str {
        let s: &'a str = *self; s
    }

    #[inline]
    fn into_owned(self) -> ~str { self.to_owned() }
}

impl<'a> Container for &'a str {
    #[inline]
    fn len(&self) -> uint {
        self.repr().len
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
            self.set_len(0)
        }
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
    /// let v: ~[char] = "abc åäö".chars().collect();
    /// assert_eq!(v, ~['a', 'b', 'c', ' ', 'å', 'ä', 'ö']);
    /// ```
    fn chars(&self) -> CharIterator<'a>;

    /// An iterator over the characters of `self`, in reverse order.
    fn chars_rev(&self) -> CharRevIterator<'a>;

    /// An iterator over the bytes of `self`
    fn bytes(&self) -> ByteIterator<'a>;

    /// An iterator over the bytes of `self`, in reverse order
    fn bytes_rev(&self) -> ByteRevIterator<'a>;

    /// An iterator over the characters of `self` and their byte offsets.
    fn char_indices(&self) -> CharOffsetIterator<'a>;

    /// An iterator over the characters of `self` and their byte offsets,
    /// in reverse order.
    fn char_indices_rev(&self) -> CharOffsetRevIterator<'a>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[&str] = "Mary had a little lamb".split(' ').collect();
    /// assert_eq!(v, ~["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: ~[&str] = "abc1def2ghi".split(|c: char| c.is_digit()).collect();
    /// assert_eq!(v, ~["abc", "def", "ghi"]);
    ///
    /// let v: ~[&str] = "lionXXtigerXleopard".split('X').collect();
    /// assert_eq!(v, ~["lion", "", "tiger", "leopard"]);
    /// ```
    fn split<Sep: CharEq>(&self, sep: Sep) -> CharSplitIterator<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, restricted to splitting at most `count`
    /// times.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[&str] = "Mary had a little lambda".splitn(' ', 2).collect();
    /// assert_eq!(v, ~["Mary", "had", "a little lambda"]);
    ///
    /// let v: ~[&str] = "abc1def2ghi".splitn(|c: char| c.is_digit(), 1).collect();
    /// assert_eq!(v, ~["abc", "def2ghi"]);
    ///
    /// let v: ~[&str] = "lionXXtigerXleopard".splitn('X', 2).collect();
    /// assert_eq!(v, ~["lion", "", "tigerXleopard"]);
    /// ```
    fn splitn<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitNIterator<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`.
    ///
    /// Equivalent to `split`, except that the trailing substring
    /// is skipped if empty (terminator semantics).
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[&str] = "A.B.".split_terminator('.').collect();
    /// assert_eq!(v, ~["A", "B"]);
    ///
    /// let v: ~[&str] = "A..B..".split_terminator('.').collect();
    /// assert_eq!(v, ~["A", "", "B", ""]);
    /// ```
    fn split_terminator<Sep: CharEq>(&self, sep: Sep) -> CharSplitIterator<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, in reverse order.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[&str] = "Mary had a little lamb".rsplit(' ').collect();
    /// assert_eq!(v, ~["lamb", "little", "a", "had", "Mary"]);
    ///
    /// let v: ~[&str] = "abc1def2ghi".rsplit(|c: char| c.is_digit()).collect();
    /// assert_eq!(v, ~["ghi", "def", "abc"]);
    ///
    /// let v: ~[&str] = "lionXXtigerXleopard".rsplit('X').collect();
    /// assert_eq!(v, ~["leopard", "tiger", "", "lion"]);
    /// ```
    fn rsplit<Sep: CharEq>(&self, sep: Sep) -> CharRSplitIterator<'a, Sep>;

    /// An iterator over substrings of `self`, separated by characters
    /// matched by `sep`, starting from the end of the string.
    /// Restricted to splitting at most `count` times.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[&str] = "Mary had a little lamb".rsplitn(' ', 2).collect();
    /// assert_eq!(v, ~["lamb", "little", "Mary had a"]);
    ///
    /// let v: ~[&str] = "abc1def2ghi".rsplitn(|c: char| c.is_digit(), 1).collect();
    /// assert_eq!(v, ~["ghi", "abc1def"]);
    ///
    /// let v: ~[&str] = "lionXXtigerXleopard".rsplitn('X', 2).collect();
    /// assert_eq!(v, ~["leopard", "tiger", "lionX"]);
    /// ```
    fn rsplitn<Sep: CharEq>(&self, sep: Sep, count: uint) -> CharSplitNIterator<'a, Sep>;

    /// An iterator over the start and end indices of the disjoint
    /// matches of `sep` within `self`.
    ///
    /// That is, each returned value `(start, end)` satisfies
    /// `self.slice(start, end) == sep`. For matches of `sep` within
    /// `self` that overlap, only the indicies corresponding to the
    /// first match are returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[(uint, uint)] = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, ~[(0,3), (6,9), (12,15)]);
    ///
    /// let v: ~[(uint, uint)] = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, ~[(1,4), (4,7)]);
    ///
    /// let v: ~[(uint, uint)] = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, ~[(0, 3)]); // only the first `aba`
    /// ```
    fn match_indices(&self, sep: &'a str) -> MatchesIndexIterator<'a>;

    /// An iterator over the substrings of `self` separated by `sep`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let v: ~[&str] = "abcXXXabcYYYabc".split_str("abc").collect();
    /// assert_eq!(v, ~["", "XXX", "YYY", ""]);
    ///
    /// let v: ~[&str] = "1abcabc2".split_str("abc").collect();
    /// assert_eq!(v, ~["1", "", "2"]);
    /// ```
    fn split_str(&self, &'a str) -> StrSplitIterator<'a>;

    /// An iterator over the lines of a string (subsequences separated
    /// by `\n`). This does not include the empty string after a
    /// trailing `\n`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let four_lines = "foo\nbar\n\nbaz\n";
    /// let v: ~[&str] = four_lines.lines().collect();
    /// assert_eq!(v, ~["foo", "bar", "", "baz"]);
    /// ```
    fn lines(&self) -> CharSplitIterator<'a, char>;

    /// An iterator over the lines of a string, separated by either
    /// `\n` or `\r\n`. As with `.lines()`, this does not include an
    /// empty trailing line.
    ///
    /// # Example
    ///
    /// ```rust
    /// let four_lines = "foo\r\nbar\n\r\nbaz\n";
    /// let v: ~[&str] = four_lines.lines_any().collect();
    /// assert_eq!(v, ~["foo", "bar", "", "baz"]);
    /// ```
    fn lines_any(&self) -> AnyLineIterator<'a>;

    /// An iterator over the words of a string (subsequences separated
    /// by any sequence of whitespace). Sequences of whitespace are
    /// collapsed, so empty "words" are not included.
    ///
    /// # Example
    ///
    /// ```rust
    /// let some_words = " Mary   had\ta little  \n\t lamb";
    /// let v: ~[&str] = some_words.words().collect();
    /// assert_eq!(v, ~["Mary", "had", "a", "little", "lamb"]);
    /// ```
    fn words(&self) -> WordIterator<'a>;

    /// An Iterator over the string in Unicode Normalization Form D
    /// (canonical decomposition).
    fn nfd_chars(&self) -> NormalizationIterator<'a>;

    /// An Iterator over the string in Unicode Normalization Form KD
    /// (compatibility decomposition).
    fn nfkd_chars(&self) -> NormalizationIterator<'a>;

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
    /// println(c);
    /// println(d);
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
    /// Due to the design of UTF-8, this operation is `O(end -
    /// begin)`. See `slice`, `slice_to` and `slice_from` for `O(1)`
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

    /// Escape each char in `s` with `char::escape_default`.
    fn escape_default(&self) -> ~str;

    /// Escape each char in `s` with `char::escape_unicode`.
    fn escape_unicode(&self) -> ~str;

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
    /// assert_eq!("11foo1bar11".trim_chars(&'1'), "foo1bar")
    /// assert_eq!("12foo1bar12".trim_chars(& &['1', '2']), "foo1bar")
    /// assert_eq!("123foo1bar123".trim_chars(&|c: char| c.is_digit()), "foo1bar")
    /// ```
    fn trim_chars<C: CharEq>(&self, to_trim: &C) -> &'a str;

    /// Returns a string with leading `chars_to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_left_chars(&'1'), "foo1bar11")
    /// assert_eq!("12foo1bar12".trim_left_chars(& &['1', '2']), "foo1bar12")
    /// assert_eq!("123foo1bar123".trim_left_chars(&|c: char| c.is_digit()), "foo1bar123")
    /// ```
    fn trim_left_chars<C: CharEq>(&self, to_trim: &C) -> &'a str;

    /// Returns a string with trailing `chars_to_trim` removed.
    ///
    /// # Arguments
    ///
    /// * to_trim - a character matcher
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!("11foo1bar11".trim_right_chars(&'1'), "11foo1bar")
    /// assert_eq!("12foo1bar12".trim_right_chars(& &['1', '2']), "12foo1bar")
    /// assert_eq!("123foo1bar123".trim_right_chars(&|c: char| c.is_digit()), "123foo1bar")
    /// ```
    fn trim_right_chars<C: CharEq>(&self, to_trim: &C) -> &'a str;

    /// Replace all occurrences of one string with another.
    ///
    /// # Arguments
    ///
    /// * `from` - The string to replace
    /// * `to` - The replacement string
    ///
    /// # Return value
    ///
    /// The original string with all occurances of `from` replaced with `to`.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = ~"Do you know the muffin man,
    /// The muffin man, the muffin man, ...";
    ///
    /// assert_eq!(s.replace("muffin man", "little lamb"),
    ///            ~"Do you know the little lamb,
    /// The little lamb, the little lamb, ...");
    ///
    /// // not found, so no change.
    /// assert_eq!(s.replace("cookie monster", "little lamb"), s);
    /// ```
    fn replace(&self, from: &str, to: &str) -> ~str;

    /// Copy a slice into a new owned str.
    fn to_owned(&self) -> ~str;

    /// Copy a slice into a new managed str.
    fn to_managed(&self) -> @str;

    /// Converts to a vector of `u16` encoded as UTF-16.
    fn to_utf16(&self) -> ~[u16];

    /// Copy a slice into a new `SendStr`.
    fn to_send_str(&self) -> SendStr;

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
    /// ```
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
    fn char_range_at_reverse(&self, start: uint) -> CharRange;

    /// Plucks the character starting at the `i`th byte of a string
    fn char_at(&self, i: uint) -> char;

    /// Plucks the character ending at the `i`th byte of a string
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

    /// Given a string, make a new string with repeated copies of it.
    fn repeat(&self, nn: uint) -> ~str;

    /// Retrieves the first character from a string slice and returns
    /// it. This does not allocate a new string; instead, it returns a
    /// slice that point one character beyond the character that was
    /// shifted.
    ///
    /// # Failure
    ///
    /// If the string does not contain any characters.
    ///
    /// # Example
    ///
    /// ```rust
    /// let s = "Löwe 老虎 Léopard";
    /// let (c, s1) = s.slice_shift_char();
    /// assert_eq!(c, 'L');
    /// assert_eq!(s1, "öwe 老虎 Léopard");
    ///
    /// let (c, s2) = s1.slice_shift_char();
    /// assert_eq!(c, 'ö');
    /// assert_eq!(s2, "we 老虎 Léopard");
    /// ```
    fn slice_shift_char(&self) -> (char, &'a str);

    /// Levenshtein Distance between two strings.
    fn lev_distance(&self, t: &str) -> uint;

    /// Returns the byte offset of an inner slice relative to an enclosing outer slice.
    ///
    /// Fails if `inner` is not a direct slice contained within self.
    ///
    /// # Example
    ///
    /// ```rust
    /// let string = "a\nb\nc";
    /// let lines: ~[&str] = string.lines().collect();
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
    fn as_ptr(&self) -> *u8;
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
    fn chars(&self) -> CharIterator<'a> {
        CharIterator{string: *self}
    }

    #[inline]
    fn chars_rev(&self) -> CharRevIterator<'a> {
        self.chars().invert()
    }

    #[inline]
    fn bytes(&self) -> ByteIterator<'a> {
        self.as_bytes().iter().map(|&b| b)
    }

    #[inline]
    fn bytes_rev(&self) -> ByteRevIterator<'a> {
        self.bytes().invert()
    }

    #[inline]
    fn char_indices(&self) -> CharOffsetIterator<'a> {
        CharOffsetIterator{string: *self, iter: self.chars()}
    }

    #[inline]
    fn char_indices_rev(&self) -> CharOffsetRevIterator<'a> {
        self.char_indices().invert()
    }

    #[inline]
    fn split<Sep: CharEq>(&self, sep: Sep) -> CharSplitIterator<'a, Sep> {
        CharSplitIterator {
            string: *self,
            only_ascii: sep.only_ascii(),
            sep: sep,
            allow_trailing_empty: true,
            finished: false,
        }
    }

    #[inline]
    fn splitn<Sep: CharEq>(&self, sep: Sep, count: uint)
        -> CharSplitNIterator<'a, Sep> {
        CharSplitNIterator {
            iter: self.split(sep),
            count: count,
            invert: false,
        }
    }

    #[inline]
    fn split_terminator<Sep: CharEq>(&self, sep: Sep)
        -> CharSplitIterator<'a, Sep> {
        CharSplitIterator {
            allow_trailing_empty: false,
            ..self.split(sep)
        }
    }

    #[inline]
    fn rsplit<Sep: CharEq>(&self, sep: Sep) -> CharRSplitIterator<'a, Sep> {
        self.split(sep).invert()
    }

    #[inline]
    fn rsplitn<Sep: CharEq>(&self, sep: Sep, count: uint)
        -> CharSplitNIterator<'a, Sep> {
        CharSplitNIterator {
            iter: self.split(sep),
            count: count,
            invert: true,
        }
    }

    #[inline]
    fn match_indices(&self, sep: &'a str) -> MatchesIndexIterator<'a> {
        assert!(!sep.is_empty())
        MatchesIndexIterator {
            haystack: *self,
            needle: sep,
            position: 0
        }
    }

    #[inline]
    fn split_str(&self, sep: &'a str) -> StrSplitIterator<'a> {
        StrSplitIterator {
            it: self.match_indices(sep),
            last_end: 0,
            finished: false
        }
    }

    #[inline]
    fn lines(&self) -> CharSplitIterator<'a, char> {
        self.split_terminator('\n')
    }

    fn lines_any(&self) -> AnyLineIterator<'a> {
        self.lines().map(|line| {
            let l = line.len();
            if l > 0 && line[l - 1] == '\r' as u8 { line.slice(0, l - 1) }
            else { line }
        })
    }

    #[inline]
    fn words(&self) -> WordIterator<'a> {
        self.split(char::is_whitespace).filter(|s| !s.is_empty())
    }

    #[inline]
    fn nfd_chars(&self) -> NormalizationIterator<'a> {
        NormalizationIterator {
            iter: self.chars(),
            buffer: ~[],
            sorted: false,
            kind: NFD
        }
    }

    #[inline]
    fn nfkd_chars(&self) -> NormalizationIterator<'a> {
        NormalizationIterator {
            iter: self.chars(),
            buffer: ~[],
            sorted: false,
            kind: NFKD
        }
    }

    #[inline]
    fn is_whitespace(&self) -> bool { self.chars().all(char::is_whitespace) }

    #[inline]
    fn is_alphanumeric(&self) -> bool { self.chars().all(char::is_alphanumeric) }

    #[inline]
    fn char_len(&self) -> uint { self.chars().len() }

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

    fn escape_default(&self) -> ~str {
        let mut out: ~str = ~"";
        out.reserve_at_least(self.len());
        for c in self.chars() {
            c.escape_default(|c| out.push_char(c));
        }
        out
    }

    fn escape_unicode(&self) -> ~str {
        let mut out: ~str = ~"";
        out.reserve_at_least(self.len());
        for c in self.chars() {
            c.escape_unicode(|c| out.push_char(c));
        }
        out
    }

    #[inline]
    fn trim(&self) -> &'a str {
        self.trim_left().trim_right()
    }

    #[inline]
    fn trim_left(&self) -> &'a str {
        self.trim_left_chars(&char::is_whitespace)
    }

    #[inline]
    fn trim_right(&self) -> &'a str {
        self.trim_right_chars(&char::is_whitespace)
    }

    #[inline]
    fn trim_chars<C: CharEq>(&self, to_trim: &C) -> &'a str {
        self.trim_left_chars(to_trim).trim_right_chars(to_trim)
    }

    #[inline]
    fn trim_left_chars<C: CharEq>(&self, to_trim: &C) -> &'a str {
        match self.find(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(first) => unsafe { raw::slice_bytes(*self, first, self.len()) }
        }
    }

    #[inline]
    fn trim_right_chars<C: CharEq>(&self, to_trim: &C) -> &'a str {
        match self.rfind(|c: char| !to_trim.matches(c)) {
            None => "",
            Some(last) => {
                let next = self.char_range_at(last).next;
                unsafe { raw::slice_bytes(*self, 0u, next) }
            }
        }
    }

    fn replace(&self, from: &str, to: &str) -> ~str {
        let mut result = ~"";
        let mut last_end = 0;
        for (start, end) in self.match_indices(from) {
            result.push_str(unsafe{raw::slice_bytes(*self, last_end, start)});
            result.push_str(to);
            last_end = end;
        }
        result.push_str(unsafe{raw::slice_bytes(*self, last_end, self.len())});
        result
    }

    #[inline]
    fn to_owned(&self) -> ~str {
        let len = self.len();
        unsafe {
            let mut v = vec::with_capacity(len);

            ptr::copy_memory(v.as_mut_ptr(), self.as_ptr(), len);
            v.set_len(len);
            ::cast::transmute(v)
        }
    }

    #[inline]
    fn to_managed(&self) -> @str {
        unsafe {
            let v: *&[u8] = cast::transmute(self);
            cast::transmute(at_vec::to_managed(*v))
        }
    }

    fn to_utf16(&self) -> ~[u16] {
        let mut u = ~[];
        for ch in self.chars() {
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

    #[inline]
    fn to_send_str(&self) -> SendStr {
        SendStrOwned(self.to_owned())
    }

    #[inline]
    fn is_char_boundary(&self, index: uint) -> bool {
        if index == self.len() { return true; }
        let b = self[index];
        return b < 128u8 || b >= 192u8;
    }

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

            return CharRange {ch: unsafe { transmute(val as u32) }, next: i + w};
        }

        return multibyte_char_range_at(*self, i);
    }

    #[inline]
    fn char_at(&self, i: uint) -> char { self.char_range_at(i).ch }

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

            let mut val = s[i] as uint;
            let w = UTF8_CHAR_WIDTH[val] as uint;
            assert!((w != 0));

            val = utf8_first_byte!(val, w);
            val = utf8_acc_cont_byte!(val, s[i + 1]);
            if w > 2 { val = utf8_acc_cont_byte!(val, s[i + 2]); }
            if w > 3 { val = utf8_acc_cont_byte!(val, s[i + 3]); }

            return CharRange {ch: unsafe { transmute(val as u32) }, next: i};
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
        unsafe { cast::transmute(*self) }
    }

    fn find<C: CharEq>(&self, search: C) -> Option<uint> {
        if search.only_ascii() {
            self.bytes().position(|b| search.matches(b as char))
        } else {
            for (index, c) in self.char_indices() {
                if search.matches(c) { return Some(index); }
            }
            None
        }
    }

    fn rfind<C: CharEq>(&self, search: C) -> Option<uint> {
        if search.only_ascii() {
            self.bytes().rposition(|b| search.matches(b as char))
        } else {
            for (index, c) in self.char_indices_rev() {
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

    fn repeat(&self, nn: uint) -> ~str {
        let mut ret = with_capacity(nn * self.len());
        for _ in range(0, nn) {
            ret.push_str(*self);
        }
        ret
    }

    #[inline]
    fn slice_shift_char(&self) -> (char, &'a str) {
        let CharRange {ch, next} = self.char_range_at(0u);
        let next_s = unsafe { raw::slice_bytes(*self, next, self.len()) };
        return (ch, next_s);
    }

    fn lev_distance(&self, t: &str) -> uint {
        let slen = self.len();
        let tlen = t.len();

        if slen == 0 { return tlen; }
        if tlen == 0 { return slen; }

        let mut dcol = vec::from_fn(tlen + 1, |x| x);

        for (i, sc) in self.chars().enumerate() {

            let mut current = i;
            dcol[0] = current + 1;

            for (j, tc) in t.chars().enumerate() {

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
    fn as_ptr(&self) -> *u8 {
        self.repr().data
    }
}

/// Methods for owned strings
pub trait OwnedStr {
    /// Appends a string slice to the back of a string, without overallocating.
    fn push_str_no_overallocate(&mut self, rhs: &str);

    /// Appends a string slice to the back of a string
    fn push_str(&mut self, rhs: &str);

    /// Appends a character to the back of a string
    fn push_char(&mut self, c: char);

    /// Remove the final character from a string and return it
    ///
    /// # Failure
    ///
    /// If the string does not contain any characters
    fn pop_char(&mut self) -> char;

    /// Remove the first character from a string and return it
    ///
    /// # Failure
    ///
    /// If the string does not contain any characters
    fn shift_char(&mut self) -> char;

    /// Prepend a char to a string
    fn unshift_char(&mut self, ch: char);

    /// Insert a new sub-string at the given position in a string, in O(n + m) time
    /// (with n and m the lengths of the string and the substring.)
    /// This fails if `position` is not at a character boundary.
    fn insert(&mut self, position: uint, substring: &str);

    /// Insert a char at the given position in a string, in O(n + m) time
    /// (with n and m the lengths of the string and the substring.)
    /// This fails if `position` is not at a character boundary.
    fn insert_char(&mut self, position: uint, ch: char);

    /// Concatenate two strings together.
    fn append(self, rhs: &str) -> ~str;

    /// Reserves capacity for exactly `n` bytes in the given string.
    ///
    /// Assuming single-byte characters, the resulting string will be large
    /// enough to hold a string of length `n`.
    ///
    /// If the capacity for `s` is already equal to or greater than the requested
    /// capacity, then no action is taken.
    ///
    /// # Arguments
    ///
    /// * s - A string
    /// * n - The number of bytes to reserve space for
    fn reserve(&mut self, n: uint);

    /// Reserves capacity for at least `n` bytes in the given string.
    ///
    /// Assuming single-byte characters, the resulting string will be large
    /// enough to hold a string of length `n`.
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
    fn reserve_at_least(&mut self, n: uint);

    /// Returns the number of single-byte characters the string can hold without
    /// reallocating
    fn capacity(&self) -> uint;

    /// Shorten a string to the specified length (which must be <= the current length)
    fn truncate(&mut self, len: uint);

    /// Consumes the string, returning the underlying byte buffer.
    ///
    /// The buffer does not have a null terminator.
    fn into_bytes(self) -> ~[u8];

    /// Sets the length of a string
    ///
    /// This will explicitly set the size of the string, without actually
    /// modifying its buffers, so it is up to the caller to ensure that
    /// the string is actually the specified size.
    unsafe fn set_len(&mut self, new_len: uint);
}

impl OwnedStr for ~str {
    #[inline]
    fn push_str_no_overallocate(&mut self, rhs: &str) {
        let new_cap = self.len() + rhs.len();
        self.reserve(new_cap);
        self.push_str(rhs);
    }

    #[inline]
    fn push_str(&mut self, rhs: &str) {
        unsafe {
            raw::push_bytes(self, rhs.as_bytes());
        }
    }

    #[inline]
    fn push_char(&mut self, c: char) {
        let cur_len = self.len();
        // may use up to 4 bytes.
        unsafe {
            let v = raw::as_owned_vec(self);
            v.reserve_additional(4);

            // Attempt to not use an intermediate buffer by just pushing bytes
            // directly onto this string.
            let write_ptr = v.as_mut_ptr().offset(cur_len as int);
            let used = vec::raw::mut_buf_as_slice(write_ptr, 4, |slc| c.encode_utf8(slc));

            v.set_len(cur_len + used);
        }
    }

    #[inline]
    fn pop_char(&mut self) -> char {
        let end = self.len();
        assert!(end > 0u);
        let CharRange {ch, next} = self.char_range_at_reverse(end);
        unsafe { self.set_len(next); }
        return ch;
    }

    #[inline]
    fn shift_char(&mut self) -> char {
        let CharRange {ch, next} = self.char_range_at(0u);
        *self = self.slice(next, self.len()).to_owned();
        return ch;
    }

    #[inline]
    fn unshift_char(&mut self, ch: char) {
        // This could be more efficient.
        let mut new_str = ~"";
        new_str.push_char(ch);
        new_str.push_str(*self);
        *self = new_str;
    }

    #[inline]
    fn insert(&mut self, position: uint, substring: &str) {
        // This could be more efficient.
        let mut new_str = self.slice_to(position).to_owned();
        new_str.push_str(substring);
        new_str.push_str(self.slice_from(position));
        *self = new_str;
    }

    #[inline]
    fn insert_char(&mut self, position: uint, ch: char) {
        // This could be more efficient.
        let mut new_str = self.slice_to(position).to_owned();
        new_str.push_char(ch);
        new_str.push_str(self.slice_from(position));
        *self = new_str;
    }

    #[inline]
    fn append(self, rhs: &str) -> ~str {
        let mut new_str = self;
        new_str.push_str_no_overallocate(rhs);
        new_str
    }

    #[inline]
    fn reserve(&mut self, n: uint) {
        unsafe {
            raw::as_owned_vec(self).reserve(n)
        }
    }

    #[inline]
    fn reserve_at_least(&mut self, n: uint) {
        self.reserve(uint::next_power_of_two_opt(n).unwrap_or(n))
    }

    #[inline]
    fn capacity(&self) -> uint {
        unsafe {
            let buf: &~[u8] = cast::transmute(self);
            buf.capacity()
        }
    }

    #[inline]
    fn truncate(&mut self, len: uint) {
        assert!(len <= self.len());
        assert!(self.is_char_boundary(len));
        unsafe { self.set_len(len); }
    }

    #[inline]
    fn into_bytes(self) -> ~[u8] {
        unsafe { cast::transmute(self) }
    }

    #[inline]
    unsafe fn set_len(&mut self, new_len: uint) {
        raw::as_owned_vec(self).set_len(new_len)
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
impl<'a> Default for &'a str {
    fn default() -> &'a str { "" }
}

impl Default for ~str {
    fn default() -> ~str { ~"" }
}

impl Default for @str {
    fn default() -> @str { @"" }
}

#[cfg(test)]
mod tests {
    use iter::AdditiveIterator;
    use prelude::*;
    use ptr;
    use str::*;
    use send_str::{SendStrOwned, SendStrStatic};

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
    fn test_insert_char() {
        let mut data = ~"ประเทศไทย中";
        data.insert_char(15, '华');
        assert_eq!(~"ประเท华ศไทย中", data);
    }

    #[test]
    fn test_insert() {
        let mut data = ~"ประเทศไทย中";
        data.insert(15, "华中");
        assert_eq!(~"ประเท华中ศไทย中", data);
    }

    #[test]
    fn test_collect() {
        let empty = ~"";
        let s: ~str = empty.chars().collect();
        assert_eq!(empty, s);
        let data = ~"ประเทศไทย中";
        let s: ~str = data.chars().collect();
        assert_eq!(data, s);
    }

    #[test]
    fn test_extend() {
        let data = ~"ประเทศไทย中";
        let mut cpy = data.clone();
        let other = "abc";
        let mut it = other.chars();
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
    fn test_into_bytes() {
        let data = ~"asdf";
        let buf = data.into_bytes();
        assert_eq!(bytes!("asdf"), buf.as_slice());
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
        t("", "", 0);
        t("hello", "llo", 2);
        t("hello", "el", 1);
        t("αβλ", "β", 1);
        t("αβλ", "", 3);
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
    fn test_raw_from_c_str() {
        unsafe {
            let a = ~[65, 65, 65, 65, 65, 65, 65, 0];
            let b = a.as_ptr();
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
    #[should_fail]
    fn test_as_bytes_fail() {
        // Don't double free. (I'm not sure if this exercises the
        // original problem code path anymore.)
        let s = ~"";
        let _bytes = s.as_bytes();
        fail!();
    }

    #[test]
    fn test_as_ptr() {
        let buf = "hello".as_ptr();
        unsafe {
            assert_eq!(*ptr::offset(buf, 0), 'h' as u8);
            assert_eq!(*ptr::offset(buf, 1), 'e' as u8);
            assert_eq!(*ptr::offset(buf, 2), 'l' as u8);
            assert_eq!(*ptr::offset(buf, 3), 'l' as u8);
            assert_eq!(*ptr::offset(buf, 4), 'o' as u8);
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
        for line in string.lines() { lines.push(line) }
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
        let s2: ~str = from_utf8(v).to_owned();
        let mut i: uint = 0u;
        let n1: uint = s1.len();
        let n2: uint = v.len();
        assert_eq!(n1, n2);
        while i < n1 {
            let a: u8 = s1[i];
            let b: u8 = s2[i];
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
        use iter::*;
        let s = ~"ศไทย中华Việt Nam";
        let v = ~['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m'];

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
        use iter::*;
        let s = ~"ศไทย中华Việt Nam";
        let v = ~['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

        let mut pos = 0;
        let mut it = s.chars_rev();

        for c in it {
            assert_eq!(c, v[pos]);
            pos += 1;
        }
        assert_eq!(pos, v.len());
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
        let s = ~"ศไทย中华Việt Nam";
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
        let s = ~"ศไทย中华Việt Nam";
        let v = [
            224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228,
            184, 173, 229, 141, 142, 86, 105, 225, 187, 135, 116, 32, 78, 97,
            109
        ];
        let mut pos = v.len();

        for b in s.bytes_rev() {
            pos -= 1;
            assert_eq!(b, v[pos]);
        }
    }

    #[test]
    fn test_char_indicesator() {
        use iter::*;
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
        use iter::*;
        let s = "ศไทย中华Việt Nam";
        let p = [27, 26, 25, 24, 23, 20, 19, 18, 15, 12, 9, 6, 3, 0];
        let v = ['m', 'a', 'N', ' ', 't', 'ệ','i','V','华','中','ย','ท','ไ','ศ'];

        let mut pos = 0;
        let mut it = s.char_indices_rev();

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

        let split: ~[&str] = data.split(' ').collect();
        assert_eq!( split, ~["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let mut rsplit: ~[&str] = data.rsplit(' ').collect();
        rsplit.reverse();
        assert_eq!(rsplit, ~["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let split: ~[&str] = data.split(|c: char| c == ' ').collect();
        assert_eq!( split, ~["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        let mut rsplit: ~[&str] = data.rsplit(|c: char| c == ' ').collect();
        rsplit.reverse();
        assert_eq!(rsplit, ~["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

        // Unicode
        let split: ~[&str] = data.split('ä').collect();
        assert_eq!( split, ~["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let mut rsplit: ~[&str] = data.rsplit('ä').collect();
        rsplit.reverse();
        assert_eq!(rsplit, ~["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let split: ~[&str] = data.split(|c: char| c == 'ä').collect();
        assert_eq!( split, ~["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

        let mut rsplit: ~[&str] = data.rsplit(|c: char| c == 'ä').collect();
        rsplit.reverse();
        assert_eq!(rsplit, ~["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);
    }

    #[test]
    fn test_splitn_char_iterator() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: ~[&str] = data.splitn(' ', 3).collect();
        assert_eq!(split, ~["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

        let split: ~[&str] = data.splitn(|c: char| c == ' ', 3).collect();
        assert_eq!(split, ~["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

        // Unicode
        let split: ~[&str] = data.splitn('ä', 3).collect();
        assert_eq!(split, ~["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);

        let split: ~[&str] = data.splitn(|c: char| c == 'ä', 3).collect();
        assert_eq!(split, ~["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);
    }

    #[test]
    fn test_rsplitn_char_iterator() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let mut split: ~[&str] = data.rsplitn(' ', 3).collect();
        split.reverse();
        assert_eq!(split, ~["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

        let mut split: ~[&str] = data.rsplitn(|c: char| c == ' ', 3).collect();
        split.reverse();
        assert_eq!(split, ~["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

        // Unicode
        let mut split: ~[&str] = data.rsplitn('ä', 3).collect();
        split.reverse();
        assert_eq!(split, ~["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);

        let mut split: ~[&str] = data.rsplitn(|c: char| c == 'ä', 3).collect();
        split.reverse();
        assert_eq!(split, ~["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);
    }

    #[test]
    fn test_split_char_iterator_no_trailing() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let split: ~[&str] = data.split('\n').collect();
        assert_eq!(split, ~["", "Märy häd ä little lämb", "Little lämb", ""]);

        let split: ~[&str] = data.split_terminator('\n').collect();
        assert_eq!(split, ~["", "Märy häd ä little lämb", "Little lämb"]);
    }

    #[test]
    fn test_rev_split_char_iterator_no_trailing() {
        let data = "\nMäry häd ä little lämb\nLittle lämb\n";

        let mut split: ~[&str] = data.split('\n').invert().collect();
        split.reverse();
        assert_eq!(split, ~["", "Märy häd ä little lämb", "Little lämb", ""]);

        let mut split: ~[&str] = data.split_terminator('\n').invert().collect();
        split.reverse();
        assert_eq!(split, ~["", "Märy häd ä little lämb", "Little lämb"]);
    }

    #[test]
    fn test_words() {
        let data = "\n \tMäry   häd\tä  little lämb\nLittle lämb\n";
        let words: ~[&str] = data.words().collect();
        assert_eq!(words, ~["Märy", "häd", "ä", "little", "lämb", "Little", "lämb"])
    }

    #[test]
    fn test_nfd_chars() {
        assert_eq!("abc".nfd_chars().collect::<~str>(), ~"abc");
        assert_eq!("\u1e0b\u01c4".nfd_chars().collect::<~str>(), ~"d\u0307\u01c4");
        assert_eq!("\u2026".nfd_chars().collect::<~str>(), ~"\u2026");
        assert_eq!("\u2126".nfd_chars().collect::<~str>(), ~"\u03a9");
        assert_eq!("\u1e0b\u0323".nfd_chars().collect::<~str>(), ~"d\u0323\u0307");
        assert_eq!("\u1e0d\u0307".nfd_chars().collect::<~str>(), ~"d\u0323\u0307");
        assert_eq!("a\u0301".nfd_chars().collect::<~str>(), ~"a\u0301");
        assert_eq!("\u0301a".nfd_chars().collect::<~str>(), ~"\u0301a");
        assert_eq!("\ud4db".nfd_chars().collect::<~str>(), ~"\u1111\u1171\u11b6");
        assert_eq!("\uac1c".nfd_chars().collect::<~str>(), ~"\u1100\u1162");
    }

    #[test]
    fn test_nfkd_chars() {
        assert_eq!("abc".nfkd_chars().collect::<~str>(), ~"abc");
        assert_eq!("\u1e0b\u01c4".nfkd_chars().collect::<~str>(), ~"d\u0307DZ\u030c");
        assert_eq!("\u2026".nfkd_chars().collect::<~str>(), ~"...");
        assert_eq!("\u2126".nfkd_chars().collect::<~str>(), ~"\u03a9");
        assert_eq!("\u1e0b\u0323".nfkd_chars().collect::<~str>(), ~"d\u0323\u0307");
        assert_eq!("\u1e0d\u0307".nfkd_chars().collect::<~str>(), ~"d\u0323\u0307");
        assert_eq!("a\u0301".nfkd_chars().collect::<~str>(), ~"a\u0301");
        assert_eq!("\u0301a".nfkd_chars().collect::<~str>(), ~"\u0301a");
        assert_eq!("\ud4db".nfkd_chars().collect::<~str>(), ~"\u1111\u1171\u11b6");
        assert_eq!("\uac1c".nfkd_chars().collect::<~str>(), ~"\u1100\u1162");
    }

    #[test]
    fn test_lines() {
        let data = "\nMäry häd ä little lämb\n\nLittle lämb\n";
        let lines: ~[&str] = data.lines().collect();
        assert_eq!(lines, ~["", "Märy häd ä little lämb", "", "Little lämb"]);

        let data = "\nMäry häd ä little lämb\n\nLittle lämb"; // no trailing \n
        let lines: ~[&str] = data.lines().collect();
        assert_eq!(lines, ~["", "Märy häd ä little lämb", "", "Little lämb"]);
    }

    #[test]
    fn test_split_strator() {
        fn t<'a>(s: &str, sep: &'a str, u: ~[&str]) {
            let v: ~[&str] = s.split_str(sep).collect();
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
    fn test_str_default() {
        use default::Default;
        fn t<S: Default + Str>() {
            let s: S = Default::default();
            assert_eq!(s.as_slice(), "");
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

    #[test]
    fn test_str_truncate() {
        let mut s = ~"12345";
        s.truncate(5);
        assert_eq!(s.as_slice(), "12345");
        s.truncate(3);
        assert_eq!(s.as_slice(), "123");
        s.truncate(0);
        assert_eq!(s.as_slice(), "");

        let mut s = ~"12345";
        let p = s.as_ptr();
        s.truncate(3);
        s.push_str("6");
        let p_ = s.as_ptr();
        assert_eq!(p_, p);
    }

    #[test]
    #[should_fail]
    fn test_str_truncate_invalid_len() {
        let mut s = ~"12345";
        s.truncate(6);
    }

    #[test]
    #[should_fail]
    fn test_str_truncate_split_codepoint() {
        let mut s = ~"\u00FC"; // ü
        s.truncate(1);
    }

    #[test]
    fn test_str_from_utf8() {
        let xs = bytes!("hello");
        assert_eq!(from_utf8(xs), "hello");

        let xs = bytes!("ศไทย中华Việt Nam");
        assert_eq!(from_utf8(xs), "ศไทย中华Việt Nam");
    }

    #[test]
    #[should_fail]
    fn test_str_from_utf8_invalid() {
        let xs = bytes!("hello", 0xff);
        let _ = from_utf8(xs);
    }

    #[test]
    fn test_str_from_utf8_opt() {
        let xs = bytes!("hello");
        assert_eq!(from_utf8_opt(xs), Some("hello"));

        let xs = bytes!("ศไทย中华Việt Nam");
        assert_eq!(from_utf8_opt(xs), Some("ศไทย中华Việt Nam"));

        let xs = bytes!("hello", 0xff);
        assert_eq!(from_utf8_opt(xs), None);
    }

    #[test]
    fn test_str_from_utf8_owned() {
        let xs = bytes!("hello").to_owned();
        assert_eq!(from_utf8_owned(xs), ~"hello");

        let xs = bytes!("ศไทย中华Việt Nam").to_owned();
        assert_eq!(from_utf8_owned(xs), ~"ศไทย中华Việt Nam");
    }

    #[test]
    fn test_str_from_utf8_owned_opt() {
        let xs = bytes!("hello").to_owned();
        assert_eq!(from_utf8_owned_opt(xs), Some(~"hello"));

        let xs = bytes!("ศไทย中华Việt Nam").to_owned();
        assert_eq!(from_utf8_owned_opt(xs), Some(~"ศไทย中华Việt Nam"));

        let xs = bytes!("hello", 0xff).to_owned();
        assert_eq!(from_utf8_owned_opt(xs), None);
    }

    #[test]
    fn test_to_send_str() {
        assert_eq!("abcde".to_send_str(), SendStrStatic("abcde"));
        assert_eq!("abcde".to_send_str(), SendStrOwned(~"abcde"));
    }

    #[test]
    fn test_from_str() {
      let owned: Option<~str> = from_str(&"string");
      assert_eq!(owned, Some(~"string"));
      let managed: Option<@str> = from_str(&"string");
      assert_eq!(managed, Some(@"string"));
    }
}

#[cfg(test)]
mod bench {
    use extra::test::BenchHarness;
    use super::*;
    use prelude::*;

    #[bench]
    fn char_iterator(bh: &mut BenchHarness) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.char_len();

        bh.iter(|| assert_eq!(s.chars().len(), len));
    }

    #[bench]
    fn char_iterator_ascii(bh: &mut BenchHarness) {
        let s = "Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb
        Mary had a little lamb, Little lamb";
        let len = s.char_len();

        bh.iter(|| assert_eq!(s.chars().len(), len));
    }

    #[bench]
    fn char_iterator_rev(bh: &mut BenchHarness) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.char_len();

        bh.iter(|| assert_eq!(s.chars_rev().len(), len));
    }

    #[bench]
    fn char_indicesator(bh: &mut BenchHarness) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.char_len();

        bh.iter(|| assert_eq!(s.char_indices().len(), len));
    }

    #[bench]
    fn char_indicesator_rev(bh: &mut BenchHarness) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let len = s.char_len();

        bh.iter(|| assert_eq!(s.char_indices_rev().len(), len));
    }

    #[bench]
    fn split_unicode_ascii(bh: &mut BenchHarness) {
        let s = "ประเทศไทย中华Việt Namประเทศไทย中华Việt Nam";

        bh.iter(|| assert_eq!(s.split('V').len(), 3));
    }

    #[bench]
    fn split_unicode_not_ascii(bh: &mut BenchHarness) {
        struct NotAscii(char);
        impl CharEq for NotAscii {
            fn matches(&self, c: char) -> bool {
                let NotAscii(cc) = *self;
                cc == c
            }
            fn only_ascii(&self) -> bool { false }
        }
        let s = "ประเทศไทย中华Việt Namประเทศไทย中华Việt Nam";

        bh.iter(|| assert_eq!(s.split(NotAscii('V')).len(), 3));
    }


    #[bench]
    fn split_ascii(bh: &mut BenchHarness) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').len();

        bh.iter(|| assert_eq!(s.split(' ').len(), len));
    }

    #[bench]
    fn split_not_ascii(bh: &mut BenchHarness) {
        struct NotAscii(char);
        impl CharEq for NotAscii {
            #[inline]
            fn matches(&self, c: char) -> bool {
                let NotAscii(cc) = *self;
                cc == c
            }
            fn only_ascii(&self) -> bool { false }
        }
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').len();

        bh.iter(|| assert_eq!(s.split(NotAscii(' ')).len(), len));
    }

    #[bench]
    fn split_extern_fn(bh: &mut BenchHarness) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').len();
        fn pred(c: char) -> bool { c == ' ' }

        bh.iter(|| assert_eq!(s.split(pred).len(), len));
    }

    #[bench]
    fn split_closure(bh: &mut BenchHarness) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').len();

        bh.iter(|| assert_eq!(s.split(|c: char| c == ' ').len(), len));
    }

    #[bench]
    fn split_slice(bh: &mut BenchHarness) {
        let s = "Mary had a little lamb, Little lamb, little-lamb.";
        let len = s.split(' ').len();

        bh.iter(|| assert_eq!(s.split(&[' ']).len(), len));
    }

    #[bench]
    fn is_utf8_100_ascii(bh: &mut BenchHarness) {

        let s = bytes!("Hello there, the quick brown fox jumped over the lazy dog! \
                        Lorem ipsum dolor sit amet, consectetur. ");

        assert_eq!(100, s.len());
        bh.iter(|| {
            let _ = is_utf8(s);
        });
    }

    #[bench]
    fn is_utf8_100_multibyte(bh: &mut BenchHarness) {
        let s = bytes!("𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰");
        assert_eq!(100, s.len());
        bh.iter(|| {
            let _ = is_utf8(s);
        });
    }

    #[bench]
    fn bench_with_capacity(bh: &mut BenchHarness) {
        bh.iter(|| {
            let _ = with_capacity(100);
        });
    }

    #[bench]
    fn bench_push_str(bh: &mut BenchHarness) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        bh.iter(|| {
            let mut r = ~"";
            r.push_str(s);
        });
    }

    #[bench]
    fn bench_connect(bh: &mut BenchHarness) {
        let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
        let sep = "→";
        let v = [s, s, s, s, s, s, s, s, s, s];
        bh.iter(|| {
            assert_eq!(v.connect(sep).len(), s.len() * 10 + sep.len() * 9);
        })
    }
}
