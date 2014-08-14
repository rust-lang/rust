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

#![experimental]

use fmt;
use iter::Iterator;
use mem;
use option::{Option, Some, None};
use slice::ImmutableVector;
use str::StrSlice;

/// Datatype to hold one ascii character. It wraps a `u8`, with the highest bit always zero.
#[deriving(Clone, PartialEq, PartialOrd, Ord, Eq)]
pub struct Ascii { chr: u8 }

impl Ascii {
    /// Converts an ascii character into a `u8`.
    #[inline]
    pub fn to_byte(self) -> u8 {
        self.chr
    }

    /// Converts an ascii character into a `char`.
    #[inline]
    pub fn to_char(self) -> char {
        self.chr as char
    }

    #[inline]
    #[allow(missing_doc)]
    #[deprecated="renamed to `to_lowercase`"]
    pub fn to_lower(self) -> Ascii {
        self.to_lowercase()
    }

    /// Convert to lowercase.
    #[inline]
    pub fn to_lowercase(self) -> Ascii {
        Ascii{chr: ASCII_LOWER_MAP[self.chr as uint]}
    }

    #[inline]
    #[allow(missing_doc)]
    #[deprecated="renamed to `to_uppercase`"]
    pub fn to_upper(self) -> Ascii {
        self.to_uppercase()
    }

    /// Convert to uppercase.
    #[inline]
    pub fn to_uppercase(self) -> Ascii {
        Ascii{chr: ASCII_UPPER_MAP[self.chr as uint]}
    }

    /// Compares two ascii characters of equality, ignoring case.
    #[inline]
    pub fn eq_ignore_case(self, other: Ascii) -> bool {
        ASCII_LOWER_MAP[self.chr as uint] == ASCII_LOWER_MAP[other.chr as uint]
    }

    // the following methods are like ctype, and the implementation is inspired by musl

    #[inline]
    #[allow(missing_doc)]
    #[deprecated="renamed to `is_alphabetic`"]
    pub fn is_alpha(&self) -> bool {
        self.is_alphabetic()
    }

    /// Check if the character is a letter (a-z, A-Z)
    #[inline]
    pub fn is_alphabetic(&self) -> bool {
        (self.chr >= 0x41 && self.chr <= 0x5A) || (self.chr >= 0x61 && self.chr <= 0x7A)
    }

    /// Check if the character is a number (0-9)
    #[inline]
    pub fn is_digit(&self) -> bool {
        self.chr >= 0x30 && self.chr <= 0x39
    }

    #[inline]
    #[allow(missing_doc)]
    #[deprecated="renamed to `is_alphanumeric`"]
    pub fn is_alnum(&self) -> bool {
        self.is_alphanumeric()
    }

    /// Check if the character is a letter or number
    #[inline]
    pub fn is_alphanumeric(&self) -> bool {
        self.is_alpha() || self.is_digit()
    }

    /// Check if the character is a space or horizontal tab
    #[inline]
    pub fn is_blank(&self) -> bool {
        self.chr == b' ' || self.chr == b'\t'
    }

    /// Check if the character is a control character
    #[inline]
    pub fn is_control(&self) -> bool {
        self.chr < 0x20 || self.chr == 0x7F
    }

    /// Checks if the character is printable (except space)
    #[inline]
    pub fn is_graph(&self) -> bool {
        (self.chr - 0x21) < 0x5E
    }

    /// Checks if the character is printable (including space)
    #[inline]
    pub fn is_print(&self) -> bool {
        (self.chr - 0x20) < 0x5F
    }

    #[inline]
    #[allow(missing_doc)]
    #[deprecated="renamed to `is_lowercase`"]
    pub fn is_lower(&self) -> bool {
        self.is_lowercase()
    }

    /// Checks if the character is lowercase
    #[inline]
    pub fn is_lowercase(&self) -> bool {
        (self.chr - b'a') < 26
    }

    #[inline]
    #[allow(missing_doc)]
    #[deprecated="renamed to `is_uppercase`"]
    pub fn is_upper(&self) -> bool {
        self.is_uppercase()
    }

    /// Checks if the character is uppercase
    #[inline]
    pub fn is_uppercase(&self) -> bool {
        (self.chr - b'A') < 26
    }

    /// Checks if the character is punctuation
    #[inline]
    pub fn is_punctuation(&self) -> bool {
        self.is_graph() && !self.is_alnum()
    }

    /// Checks if the character is a valid hex digit
    #[inline]
    pub fn is_hex(&self) -> bool {
        self.is_digit() || ((self.chr | 32u8) - b'a') < 6
    }
}

impl<'a> fmt::Show for Ascii {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (self.chr as char).fmt(f)
    }
}

/// Trait for converting into an ascii type.
pub trait AsciiCast<T> {
    /// Convert to an ascii type, fail on non-ASCII input.
    #[inline]
    fn to_ascii(&self) -> T {
        assert!(self.is_ascii());
        unsafe {self.to_ascii_nocheck()}
    }

    /// Convert to an ascii type, return None on non-ASCII input.
    #[inline]
    fn to_ascii_opt(&self) -> Option<T> {
        if self.is_ascii() {
            Some(unsafe { self.to_ascii_nocheck() })
        } else {
            None
        }
    }

    /// Convert to an ascii type, not doing any range asserts
    unsafe fn to_ascii_nocheck(&self) -> T;

    /// Check if convertible to ascii
    fn is_ascii(&self) -> bool;
}

impl<'a> AsciiCast<&'a [Ascii]> for &'a [u8] {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> &'a [Ascii] {
        mem::transmute(*self)
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        for b in self.iter() {
            if !b.is_ascii() { return false; }
        }
        true
    }
}

impl<'a> AsciiCast<&'a [Ascii]> for &'a str {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> &'a [Ascii] {
        mem::transmute(*self)
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        self.bytes().all(|b| b.is_ascii())
    }
}

impl AsciiCast<Ascii> for u8 {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> Ascii {
        Ascii{ chr: *self }
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        *self & 128 == 0u8
    }
}

impl AsciiCast<Ascii> for char {
    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> Ascii {
        Ascii{ chr: *self as u8 }
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        *self as u32 - ('\x7F' as u32 & *self as u32) == 0
    }
}

pub static ASCII_LOWER_MAP: [u8, ..256] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    b' ', b'!', b'"', b'#', b'$', b'%', b'&', b'\'',
    b'(', b')', b'*', b'+', b',', b'-', b'.', b'/',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b':', b';', b'<', b'=', b'>', b'?',
    b'@',

          b'a', b'b', b'c', b'd', b'e', b'f', b'g',
    b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o',
    b'p', b'q', b'r', b's', b't', b'u', b'v', b'w',
    b'x', b'y', b'z',

                      b'[', b'\\', b']', b'^', b'_',
    b'`', b'a', b'b', b'c', b'd', b'e', b'f', b'g',
    b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o',
    b'p', b'q', b'r', b's', b't', b'u', b'v', b'w',
    b'x', b'y', b'z', b'{', b'|', b'}', b'~', 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

pub static ASCII_UPPER_MAP: [u8, ..256] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    b' ', b'!', b'"', b'#', b'$', b'%', b'&', b'\'',
    b'(', b')', b'*', b'+', b',', b'-', b'.', b'/',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b':', b';', b'<', b'=', b'>', b'?',
    b'@', b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
    b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W',
    b'X', b'Y', b'Z', b'[', b'\\', b']', b'^', b'_',
    b'`',

          b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
    b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W',
    b'X', b'Y', b'Z',

                      b'{', b'|', b'}', b'~', 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];
