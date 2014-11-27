// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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

//! Operations on ASCII strings and characters

#![unstable = "unsure about placement and naming"]
#![allow(deprecated)]

use core::kinds::Sized;
use fmt;
use iter::Iterator;
use mem;
use option::{Option, Some, None};
use slice::{SlicePrelude, AsSlice};
use str::{Str, StrPrelude};
use string::{String, IntoString};
use vec::Vec;

/// Datatype to hold one ascii character. It wraps a `u8`, with the highest bit always zero.
#[deriving(Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Ascii { chr: u8 }

impl Ascii {
    /// Converts an ascii character into a `u8`.
    #[inline]
    #[unstable = "recently renamed"]
    pub fn as_byte(&self) -> u8 { unimplemented!() }

    /// Deprecated: use `as_byte` isntead.
    #[deprecated = "use as_byte"]
    pub fn to_byte(self) -> u8 { unimplemented!() }

    /// Converts an ascii character into a `char`.
    #[inline]
    #[unstable = "recently renamed"]
    pub fn as_char(&self) -> char { unimplemented!() }

    /// Deprecated: use `as_char` isntead.
    #[deprecated = "use as_char"]
    pub fn to_char(self) -> char { unimplemented!() }

    /// Convert to lowercase.
    #[inline]
    #[stable]
    pub fn to_lowercase(&self) -> Ascii { unimplemented!() }

    /// Convert to uppercase.
    #[inline]
    #[stable]
    pub fn to_uppercase(&self) -> Ascii { unimplemented!() }

    /// Compares two ascii characters of equality, ignoring case.
    #[inline]
    #[deprecated = "normalize with to_lowercase"]
    pub fn eq_ignore_case(self, other: Ascii) -> bool { unimplemented!() }

    // the following methods are like ctype, and the implementation is inspired by musl

    /// Check if the character is a letter (a-z, A-Z)
    #[inline]
    #[stable]
    pub fn is_alphabetic(&self) -> bool { unimplemented!() }

    /// Check if the character is a number (0-9)
    #[inline]
    #[unstable = "may be renamed"]
    pub fn is_digit(&self) -> bool { unimplemented!() }

    /// Check if the character is a letter or number
    #[inline]
    #[stable]
    pub fn is_alphanumeric(&self) -> bool { unimplemented!() }

    /// Check if the character is a space or horizontal tab
    #[inline]
    #[experimental = "likely to be removed"]
    pub fn is_blank(&self) -> bool { unimplemented!() }

    /// Check if the character is a control character
    #[inline]
    #[stable]
    pub fn is_control(&self) -> bool { unimplemented!() }

    /// Checks if the character is printable (except space)
    #[inline]
    #[experimental = "unsure about naming, or whether this is needed"]
    pub fn is_graph(&self) -> bool { unimplemented!() }

    /// Checks if the character is printable (including space)
    #[inline]
    #[unstable = "unsure about naming"]
    pub fn is_print(&self) -> bool { unimplemented!() }

    /// Checks if the character is alphabetic and lowercase
    #[inline]
    #[stable]
    pub fn is_lowercase(&self) -> bool { unimplemented!() }

    /// Checks if the character is alphabetic and uppercase
    #[inline]
    #[stable]
    pub fn is_uppercase(&self) -> bool { unimplemented!() }

    /// Checks if the character is punctuation
    #[inline]
    #[stable]
    pub fn is_punctuation(&self) -> bool { unimplemented!() }

    /// Checks if the character is a valid hex digit
    #[inline]
    #[stable]
    pub fn is_hex(&self) -> bool { unimplemented!() }
}

impl<'a> fmt::Show for Ascii {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

/// Trait for converting into an ascii type.
#[experimental = "may be replaced by generic conversion traits"]
pub trait AsciiCast<T> {
    /// Convert to an ascii type, panic on non-ASCII input.
    #[inline]
    fn to_ascii(&self) -> T { unimplemented!() }

    /// Convert to an ascii type, return None on non-ASCII input.
    #[inline]
    fn to_ascii_opt(&self) -> Option<T> { unimplemented!() }

    /// Convert to an ascii type, not doing any range asserts
    unsafe fn to_ascii_nocheck(&self) -> T;

    /// Check if convertible to ascii
    fn is_ascii(&self) -> bool;
}

#[experimental = "may be replaced by generic conversion traits"]
impl<'a> AsciiCast<&'a[Ascii]> for &'a [u8] {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> &'a[Ascii] { unimplemented!() }

    #[inline]
    fn is_ascii(&self) -> bool { unimplemented!() }
}

#[experimental = "may be replaced by generic conversion traits"]
impl<'a> AsciiCast<&'a [Ascii]> for &'a str {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> &'a [Ascii] { unimplemented!() }

    #[inline]
    fn is_ascii(&self) -> bool { unimplemented!() }
}

#[experimental = "may be replaced by generic conversion traits"]
impl AsciiCast<Ascii> for u8 {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> Ascii { unimplemented!() }

    #[inline]
    fn is_ascii(&self) -> bool { unimplemented!() }
}

#[experimental = "may be replaced by generic conversion traits"]
impl AsciiCast<Ascii> for char {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> Ascii { unimplemented!() }

    #[inline]
    fn is_ascii(&self) -> bool { unimplemented!() }
}

/// Trait for copyless casting to an ascii vector.
#[experimental = "may be replaced by generic conversion traits"]
pub trait OwnedAsciiCast {
    /// Check if convertible to ascii
    fn is_ascii(&self) -> bool;

    /// Take ownership and cast to an ascii vector.
    /// # Panics
    ///
    /// Panic on non-ASCII input.
    #[inline]
    fn into_ascii(self) -> Vec<Ascii> { unimplemented!() }

    /// Take ownership and cast to an ascii vector. Return None on non-ASCII input.
    #[inline]
    fn into_ascii_opt(self) -> Option<Vec<Ascii>> { unimplemented!() }

    /// Take ownership and cast to an ascii vector.
    /// Does not perform validation checks.
    unsafe fn into_ascii_nocheck(self) -> Vec<Ascii>;
}

#[experimental = "may be replaced by generic conversion traits"]
impl OwnedAsciiCast for String {
    #[inline]
    fn is_ascii(&self) -> bool { unimplemented!() }

    #[inline]
    unsafe fn into_ascii_nocheck(self) -> Vec<Ascii> { unimplemented!() }
}

#[experimental = "may be replaced by generic conversion traits"]
impl OwnedAsciiCast for Vec<u8> {
    #[inline]
    fn is_ascii(&self) -> bool { unimplemented!() }

    #[inline]
    unsafe fn into_ascii_nocheck(self) -> Vec<Ascii> { unimplemented!() }
}

/// Trait for converting an ascii type to a string. Needed to convert
/// `&[Ascii]` to `&str`.
#[experimental = "may be replaced by generic conversion traits"]
pub trait AsciiStr for Sized? {
    /// Convert to a string.
    fn as_str_ascii<'a>(&'a self) -> &'a str;

    /// Deprecated: use `to_lowercase`
    #[deprecated="renamed `to_lowercase`"]
    fn to_lower(&self) -> Vec<Ascii>;

    /// Convert to vector representing a lower cased ascii string.
    #[deprecated = "use iterators instead"]
    fn to_lowercase(&self) -> Vec<Ascii>;

    /// Deprecated: use `to_uppercase`
    #[deprecated="renamed `to_uppercase`"]
    fn to_upper(&self) -> Vec<Ascii>;

    /// Convert to vector representing a upper cased ascii string.
    #[deprecated = "use iterators instead"]
    fn to_uppercase(&self) -> Vec<Ascii>;

    /// Compares two Ascii strings ignoring case.
    #[deprecated = "use iterators instead"]
    fn eq_ignore_case(&self, other: &[Ascii]) -> bool;
}

#[experimental = "may be replaced by generic conversion traits"]
impl AsciiStr for [Ascii] {
    #[inline]
    fn as_str_ascii<'a>(&'a self) -> &'a str { unimplemented!() }

    #[inline]
    fn to_lower(&self) -> Vec<Ascii> { unimplemented!() }

    #[inline]
    fn to_lowercase(&self) -> Vec<Ascii> { unimplemented!() }

    #[inline]
    fn to_upper(&self) -> Vec<Ascii> { unimplemented!() }

    #[inline]
    fn to_uppercase(&self) -> Vec<Ascii> { unimplemented!() }

    #[inline]
    fn eq_ignore_case(&self, other: &[Ascii]) -> bool { unimplemented!() }
}

impl IntoString for Vec<Ascii> {
    #[inline]
    fn into_string(self) -> String { unimplemented!() }
}

/// Trait to convert to an owned byte vector by consuming self
#[experimental = "may be replaced by generic conversion traits"]
pub trait IntoBytes {
    /// Converts to an owned byte vector by consuming self
    fn into_bytes(self) -> Vec<u8>;
}

#[experimental = "may be replaced by generic conversion traits"]
impl IntoBytes for Vec<Ascii> {
    fn into_bytes(self) -> Vec<u8> { unimplemented!() }
}


/// Extension methods for ASCII-subset only operations on owned strings
#[experimental = "would prefer to do this in a more general way"]
pub trait OwnedAsciiExt {
    /// Convert the string to ASCII upper case:
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    fn into_ascii_upper(self) -> Self;

    /// Convert the string to ASCII lower case:
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    fn into_ascii_lower(self) -> Self;
}

/// Extension methods for ASCII-subset only operations on string slices
#[experimental = "would prefer to do this in a more general way"]
pub trait AsciiExt<T> for Sized? {
    /// Makes a copy of the string in ASCII upper case:
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    fn to_ascii_upper(&self) -> T;

    /// Makes a copy of the string in ASCII lower case:
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    fn to_ascii_lower(&self) -> T;

    /// Check that two strings are an ASCII case-insensitive match.
    /// Same as `to_ascii_lower(a) == to_ascii_lower(b)`,
    /// but without allocating and copying temporary strings.
    fn eq_ignore_ascii_case(&self, other: &Self) -> bool;
}

#[experimental = "would prefer to do this in a more general way"]
impl AsciiExt<String> for str {
    #[inline]
    fn to_ascii_upper(&self) -> String { unimplemented!() }

    #[inline]
    fn to_ascii_lower(&self) -> String { unimplemented!() }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &str) -> bool { unimplemented!() }
}

#[experimental = "would prefer to do this in a more general way"]
impl OwnedAsciiExt for String {
    #[inline]
    fn into_ascii_upper(self) -> String { unimplemented!() }

    #[inline]
    fn into_ascii_lower(self) -> String { unimplemented!() }
}

#[experimental = "would prefer to do this in a more general way"]
impl AsciiExt<Vec<u8>> for [u8] {
    #[inline]
    fn to_ascii_upper(&self) -> Vec<u8> { unimplemented!() }

    #[inline]
    fn to_ascii_lower(&self) -> Vec<u8> { unimplemented!() }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool { unimplemented!() }
}

#[experimental = "would prefer to do this in a more general way"]
impl OwnedAsciiExt for Vec<u8> {
    #[inline]
    fn into_ascii_upper(mut self) -> Vec<u8> { unimplemented!() }

    #[inline]
    fn into_ascii_lower(mut self) -> Vec<u8> { unimplemented!() }
}

/// Returns a 'default' ASCII and C++11-like literal escape of a `u8`
///
/// The default is chosen with a bias toward producing literals that are
/// legal in a variety of languages, including C++11 and similar C-family
/// languages. The exact rules are:
///
/// - Tab, CR and LF are escaped as '\t', '\r' and '\n' respectively.
/// - Single-quote, double-quote and backslash chars are backslash-escaped.
/// - Any other chars in the range [0x20,0x7e] are not escaped.
/// - Any other chars are given hex escapes.
/// - Unicode escapes are never generated by this function.
#[unstable = "needs to be updated to use an iterator"]
pub fn escape_default(c: u8, f: |u8|) { unimplemented!() }

static ASCII_LOWER_MAP: [u8, ..0] = [
];

static ASCII_UPPER_MAP: [u8, ..0] = [
];
