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

use iter::IteratorExt;
use ops::FnMut;
use slice::SliceExt;
use str::StrExt;
use string::String;
use vec::Vec;

/// Extension methods for ASCII-subset only operations on owned strings
#[unstable = "would prefer to do this in a more general way"]
pub trait OwnedAsciiExt {
    /// Convert the string to ASCII upper case:
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    fn into_ascii_uppercase(self) -> Self;

    /// Convert the string to ASCII lower case:
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    fn into_ascii_lowercase(self) -> Self;
}

/// Extension methods for ASCII-subset only operations on string slices
#[unstable = "would prefer to do this in a more general way"]
pub trait AsciiExt<T = Self> {
    /// Check if within the ASCII range.
    fn is_ascii(&self) -> bool;

    /// Makes a copy of the string in ASCII upper case:
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    fn to_ascii_uppercase(&self) -> T;

    /// Makes a copy of the string in ASCII lower case:
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    fn to_ascii_lowercase(&self) -> T;

    /// Check that two strings are an ASCII case-insensitive match.
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporary strings.
    fn eq_ignore_ascii_case(&self, other: &Self) -> bool;
}

#[unstable = "would prefer to do this in a more general way"]
impl AsciiExt<String> for str {
    #[inline]
    fn is_ascii(&self) -> bool {
        self.bytes().all(|b| b.is_ascii())
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> String {
        // Vec<u8>::to_ascii_uppercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(self.as_bytes().to_ascii_uppercase()) }
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> String {
        // Vec<u8>::to_ascii_lowercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(self.as_bytes().to_ascii_lowercase()) }
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    }
}

#[unstable = "would prefer to do this in a more general way"]
impl OwnedAsciiExt for String {
    #[inline]
    fn into_ascii_uppercase(self) -> String {
        // Vec<u8>::into_ascii_uppercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(self.into_bytes().into_ascii_uppercase()) }
    }

    #[inline]
    fn into_ascii_lowercase(self) -> String {
        // Vec<u8>::into_ascii_lowercase() preserves the UTF-8 invariant.
        unsafe { String::from_utf8_unchecked(self.into_bytes().into_ascii_lowercase()) }
    }
}

#[unstable = "would prefer to do this in a more general way"]
impl AsciiExt<Vec<u8>> for [u8] {
    #[inline]
    fn is_ascii(&self) -> bool {
        self.iter().all(|b| b.is_ascii())
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> Vec<u8> {
        self.iter().map(|b| b.to_ascii_uppercase()).collect()
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> Vec<u8> {
        self.iter().map(|b| b.to_ascii_lowercase()).collect()
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        self.len() == other.len() &&
        self.iter().zip(other.iter()).all(|(a, b)| {
            a.eq_ignore_ascii_case(b)
        })
    }
}

#[unstable = "would prefer to do this in a more general way"]
impl OwnedAsciiExt for Vec<u8> {
    #[inline]
    fn into_ascii_uppercase(mut self) -> Vec<u8> {
        for byte in self.iter_mut() {
            *byte = byte.to_ascii_uppercase();
        }
        self
    }

    #[inline]
    fn into_ascii_lowercase(mut self) -> Vec<u8> {
        for byte in self.iter_mut() {
            *byte = byte.to_ascii_lowercase();
        }
        self
    }
}

#[unstable = "would prefer to do this in a more general way"]
impl AsciiExt for u8 {
    #[inline]
    fn is_ascii(&self) -> bool {
        *self & 128 == 0u8
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> u8 {
        ASCII_UPPERCASE_MAP[*self as uint]
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> u8 {
        ASCII_LOWERCASE_MAP[*self as uint]
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &u8) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }
}

#[unstable = "would prefer to do this in a more general way"]
impl AsciiExt for char {
    #[inline]
    fn is_ascii(&self) -> bool {
        *self as u32 <= 0x7F
    }

    #[inline]
    fn to_ascii_uppercase(&self) -> char {
        if self.is_ascii() {
            (*self as u8).to_ascii_uppercase() as char
        } else {
            *self
        }
    }

    #[inline]
    fn to_ascii_lowercase(&self) -> char {
        if self.is_ascii() {
            (*self as u8).to_ascii_lowercase() as char
        } else {
            *self
        }
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &char) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }
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
pub fn escape_default<F>(c: u8, mut f: F) where
    F: FnMut(u8),
{
    match c {
        b'\t' => { f(b'\\'); f(b't'); }
        b'\r' => { f(b'\\'); f(b'r'); }
        b'\n' => { f(b'\\'); f(b'n'); }
        b'\\' => { f(b'\\'); f(b'\\'); }
        b'\'' => { f(b'\\'); f(b'\''); }
        b'"'  => { f(b'\\'); f(b'"'); }
        b'\x20' ... b'\x7e' => { f(c); }
        _ => {
            f(b'\\');
            f(b'x');
            for &offset in [4u, 0u].iter() {
                match ((c as i32) >> offset) & 0xf {
                    i @ 0 ... 9 => f(b'0' + (i as u8)),
                    i => f(b'a' + (i as u8 - 10)),
                }
            }
        }
    }
}

static ASCII_LOWERCASE_MAP: [u8; 256] = [
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

static ASCII_UPPERCASE_MAP: [u8; 256] = [
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


#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use super::*;
    use char::from_u32;

    #[test]
    fn test_ascii() {
        assert!("banana".chars().all(|c| c.is_ascii()));
        assert!(!"ประเทศไทย中华Việt Nam".chars().all(|c| c.is_ascii()));
    }

    #[test]
    fn test_ascii_vec() {
        assert!("".is_ascii());
        assert!("a".is_ascii());
        assert!(!"\u{2009}".is_ascii());

    }

    #[test]
    fn test_to_ascii_uppercase() {
        assert_eq!("url()URL()uRl()ürl".to_ascii_uppercase(), "URL()URL()URL()üRL");
        assert_eq!("hıKß".to_ascii_uppercase(), "HıKß");

        let mut i = 0;
        while i <= 500 {
            let upper = if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().to_ascii_uppercase(),
                       (from_u32(upper).unwrap()).to_string());
            i += 1;
        }
    }

    #[test]
    fn test_to_ascii_lowercase() {
        assert_eq!("url()URL()uRl()Ürl".to_ascii_lowercase(), "url()url()url()Ürl");
        // Dotted capital I, Kelvin sign, Sharp S.
        assert_eq!("HİKß".to_ascii_lowercase(), "hİKß");

        let mut i = 0;
        while i <= 500 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().to_ascii_lowercase(),
                       (from_u32(lower).unwrap()).to_string());
            i += 1;
        }
    }

    #[test]
    fn test_into_ascii_uppercase() {
        assert_eq!(("url()URL()uRl()ürl".to_string()).into_ascii_uppercase(),
                   "URL()URL()URL()üRL".to_string());
        assert_eq!(("hıKß".to_string()).into_ascii_uppercase(), "HıKß");

        let mut i = 0;
        while i <= 500 {
            let upper = if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().into_ascii_uppercase(),
                       (from_u32(upper).unwrap()).to_string());
            i += 1;
        }
    }

    #[test]
    fn test_into_ascii_lowercase() {
        assert_eq!(("url()URL()uRl()Ürl".to_string()).into_ascii_lowercase(),
                   "url()url()url()Ürl");
        // Dotted capital I, Kelvin sign, Sharp S.
        assert_eq!(("HİKß".to_string()).into_ascii_lowercase(), "hİKß");

        let mut i = 0;
        while i <= 500 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().into_ascii_lowercase(),
                       (from_u32(lower).unwrap()).to_string());
            i += 1;
        }
    }

    #[test]
    fn test_eq_ignore_ascii_case() {
        assert!("url()URL()uRl()Ürl".eq_ignore_ascii_case("url()url()url()Ürl"));
        assert!(!"Ürl".eq_ignore_ascii_case("ürl"));
        // Dotted capital I, Kelvin sign, Sharp S.
        assert!("HİKß".eq_ignore_ascii_case("hİKß"));
        assert!(!"İ".eq_ignore_ascii_case("i"));
        assert!(!"K".eq_ignore_ascii_case("k"));
        assert!(!"ß".eq_ignore_ascii_case("s"));

        let mut i = 0;
        while i <= 500 {
            let c = i;
            let lower = if 'A' as u32 <= c && c <= 'Z' as u32 { c + 'a' as u32 - 'A' as u32 }
                        else { c };
            assert!((from_u32(i).unwrap()).to_string().eq_ignore_ascii_case(
                    (from_u32(lower).unwrap()).to_string().as_slice()));
            i += 1;
        }
    }
}
