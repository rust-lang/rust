// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on ASCII strings and characters.

use to_str::{ToStr,ToStrConsume};
use str;
use str::StrSlice;
use str::OwnedStr;
use container::Container;
use cast;
use ptr;
use iter::Iterator;
use vec::{CopyableVector, ImmutableVector};
use to_bytes::IterBytes;
use option::{Some, None};

/// Datatype to hold one ascii character. It wraps a `u8`, with the highest bit always zero.
#[deriving(Clone, Eq, Ord, TotalOrd, TotalEq)]
pub struct Ascii { priv chr: u8 }

impl Ascii {
    /// Converts a ascii character into a `u8`.
    #[inline]
    pub fn to_byte(self) -> u8 {
        self.chr
    }

    /// Converts a ascii character into a `char`.
    #[inline]
    pub fn to_char(self) -> char {
        self.chr as char
    }

    /// Convert to lowercase.
    #[inline]
    pub fn to_lower(self) -> Ascii {
        Ascii{chr: ASCII_LOWER_MAP[self.chr]}
    }

    /// Convert to uppercase.
    #[inline]
    pub fn to_upper(self) -> Ascii {
        Ascii{chr: ASCII_UPPER_MAP[self.chr]}
    }

    /// Compares two ascii characters of equality, ignoring case.
    #[inline]
    pub fn eq_ignore_case(self, other: Ascii) -> bool {
        ASCII_LOWER_MAP[self.chr] == ASCII_LOWER_MAP[other.chr]
    }
}

impl ToStr for Ascii {
    #[inline]
    fn to_str(&self) -> ~str {
        // self.chr is allways a valid utf8 byte, no need for the check
        unsafe { str::raw::from_byte(self.chr) }
    }
}

/// Trait for converting into an ascii type.
pub trait AsciiCast<T> {
    /// Convert to an ascii type
    fn to_ascii(&self) -> T;

    /// Convert to an ascii type, not doing any range asserts
    unsafe fn to_ascii_nocheck(&self) -> T;

    /// Check if convertible to ascii
    fn is_ascii(&self) -> bool;
}

impl<'self> AsciiCast<&'self[Ascii]> for &'self [u8] {
    #[inline]
    fn to_ascii(&self) -> &'self[Ascii] {
        assert!(self.is_ascii());
        unsafe {self.to_ascii_nocheck()}
    }

    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> &'self[Ascii] {
        cast::transmute(*self)
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        for b in self.iter() {
            if !b.is_ascii() { return false; }
        }
        true
    }
}

impl<'self> AsciiCast<&'self [Ascii]> for &'self str {
    #[inline]
    fn to_ascii(&self) -> &'self [Ascii] {
        assert!(self.is_ascii());
        unsafe { self.to_ascii_nocheck() }
    }

    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> &'self [Ascii] {
        cast::transmute(*self)
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        self.byte_iter().all(|b| b.is_ascii())
    }
}

impl AsciiCast<Ascii> for u8 {
    #[inline]
    fn to_ascii(&self) -> Ascii {
        assert!(self.is_ascii());
        unsafe {self.to_ascii_nocheck()}
    }

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
    fn to_ascii(&self) -> Ascii {
        assert!(self.is_ascii());
        unsafe {self.to_ascii_nocheck()}
    }

    #[inline]
    unsafe fn to_ascii_nocheck(&self) -> Ascii {
        Ascii{ chr: *self as u8 }
    }

    #[inline]
    fn is_ascii(&self) -> bool {
        *self as u32 - ('\x7F' as u32 & *self as u32) == 0
    }
}

/// Trait for copyless casting to an ascii vector.
pub trait OwnedAsciiCast {
    /// Take ownership and cast to an ascii vector without trailing zero element.
    fn into_ascii(self) -> ~[Ascii];

    /// Take ownership and cast to an ascii vector without trailing zero element.
    /// Does not perform validation checks.
    unsafe fn into_ascii_nocheck(self) -> ~[Ascii];
}

impl OwnedAsciiCast for ~[u8] {
    #[inline]
    fn into_ascii(self) -> ~[Ascii] {
        assert!(self.is_ascii());
        unsafe {self.into_ascii_nocheck()}
    }

    #[inline]
    unsafe fn into_ascii_nocheck(self) -> ~[Ascii] {
        cast::transmute(self)
    }
}

impl OwnedAsciiCast for ~str {
    #[inline]
    fn into_ascii(self) -> ~[Ascii] {
        assert!(self.is_ascii());
        unsafe {self.into_ascii_nocheck()}
    }

    #[inline]
    unsafe fn into_ascii_nocheck(self) -> ~[Ascii] {
        cast::transmute(self)
    }
}

/// Trait for converting an ascii type to a string. Needed to convert `&[Ascii]` to `~str`
pub trait AsciiStr {
    /// Convert to a string.
    fn to_str_ascii(&self) -> ~str;

    /// Convert to vector representing a lower cased ascii string.
    fn to_lower(&self) -> ~[Ascii];

    /// Convert to vector representing a upper cased ascii string.
    fn to_upper(&self) -> ~[Ascii];

    /// Compares two Ascii strings ignoring case
    fn eq_ignore_case(self, other: &[Ascii]) -> bool;
}

impl<'self> AsciiStr for &'self [Ascii] {
    #[inline]
    fn to_str_ascii(&self) -> ~str {
        let cpy = self.to_owned();
        unsafe { cast::transmute(cpy) }
    }

    #[inline]
    fn to_lower(&self) -> ~[Ascii] {
        self.map(|a| a.to_lower())
    }

    #[inline]
    fn to_upper(&self) -> ~[Ascii] {
        self.map(|a| a.to_upper())
    }

    #[inline]
    fn eq_ignore_case(self, other: &[Ascii]) -> bool {
        do self.iter().zip(other.iter()).all |(&a, &b)| { a.eq_ignore_case(b) }
    }
}

impl ToStrConsume for ~[Ascii] {
    #[inline]
    fn into_str(self) -> ~str {
        unsafe { cast::transmute(self) }
    }
}

impl IterBytes for Ascii {
    #[inline]
    fn iter_bytes(&self, _lsb0: bool, f: &fn(buf: &[u8]) -> bool) -> bool {
        f([self.to_byte()])
    }
}

/// Trait to convert to a owned byte array by consuming self
pub trait ToBytesConsume {
    /// Converts to a owned byte array by consuming self
    fn into_bytes(self) -> ~[u8];
}

impl ToBytesConsume for ~[Ascii] {
    fn into_bytes(self) -> ~[u8] {
        unsafe { cast::transmute(self) }
    }
}

/// Extension methods for ASCII-subset only operations on strings
pub trait StrAsciiExt {
    /// Convert the string to ASCII upper case:
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    fn to_ascii_upper(&self) -> ~str;

    /// Convert the string to ASCII lower case:
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    fn to_ascii_lower(&self) -> ~str;

    /// Check that two strings are an ASCII case-insensitive match.
    /// Same as `to_ascii_lower(a) == to_ascii_lower(b)`,
    /// but without allocating and copying temporary strings.
    fn eq_ignore_ascii_case(&self, other: &str) -> bool;
}

impl<'self> StrAsciiExt for &'self str {
    #[inline]
    fn to_ascii_upper(&self) -> ~str {
        map_bytes(*self, ASCII_UPPER_MAP)
    }

    #[inline]
    fn to_ascii_lower(&self) -> ~str {
        map_bytes(*self, ASCII_LOWER_MAP)
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.len() == other.len() && self.as_bytes().iter().zip(other.as_bytes().iter()).all(
            |(byte_self, byte_other)| ASCII_LOWER_MAP[*byte_self] == ASCII_LOWER_MAP[*byte_other])
    }
}

#[inline]
fn map_bytes(string: &str, map: &'static [u8]) -> ~str {
    let len = string.len();
    let mut result = str::with_capacity(len);
    unsafe {
        do result.as_mut_buf |mut buf, _| {
            for c in string.as_bytes().iter() {
                *buf = map[*c];
                buf = ptr::mut_offset(buf, 1)
            }
        }
        str::raw::set_len(&mut result, len);
    }
    result
}

static ASCII_LOWER_MAP: &'static [u8] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x40, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
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

static ASCII_UPPER_MAP: &'static [u8] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
    0x60, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
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
    use super::*;
    use str::from_char;
    use char::from_u32;

    macro_rules! v2ascii (
        ( [$($e:expr),*]) => ( [$(Ascii{chr:$e}),*]);
        (~[$($e:expr),*]) => (~[$(Ascii{chr:$e}),*]);
    )

    #[test]
    fn test_ascii() {
        assert_eq!(65u8.to_ascii().to_byte(), 65u8);
        assert_eq!(65u8.to_ascii().to_char(), 'A');
        assert_eq!('A'.to_ascii().to_char(), 'A');
        assert_eq!('A'.to_ascii().to_byte(), 65u8);

        assert_eq!('A'.to_ascii().to_lower().to_char(), 'a');
        assert_eq!('Z'.to_ascii().to_lower().to_char(), 'z');
        assert_eq!('a'.to_ascii().to_upper().to_char(), 'A');
        assert_eq!('z'.to_ascii().to_upper().to_char(), 'Z');

        assert_eq!('@'.to_ascii().to_lower().to_char(), '@');
        assert_eq!('['.to_ascii().to_lower().to_char(), '[');
        assert_eq!('`'.to_ascii().to_upper().to_char(), '`');
        assert_eq!('{'.to_ascii().to_upper().to_char(), '{');

        assert!("banana".iter().all(|c| c.is_ascii()));
        assert!(!"ประเทศไทย中华Việt Nam".iter().all(|c| c.is_ascii()));
    }

    #[test]
    fn test_ascii_vec() {
        assert_eq!((&[40u8, 32u8, 59u8]).to_ascii(), v2ascii!([40, 32, 59]));
        assert_eq!("( ;".to_ascii(),                 v2ascii!([40, 32, 59]));
        // FIXME: #5475 borrowchk error, owned vectors do not live long enough
        // if chained-from directly
        let v = ~[40u8, 32u8, 59u8]; assert_eq!(v.to_ascii(), v2ascii!([40, 32, 59]));
        let v = ~"( ;";              assert_eq!(v.to_ascii(), v2ascii!([40, 32, 59]));

        assert_eq!("abCDef&?#".to_ascii().to_lower().to_str_ascii(), ~"abcdef&?#");
        assert_eq!("abCDef&?#".to_ascii().to_upper().to_str_ascii(), ~"ABCDEF&?#");

        assert_eq!("".to_ascii().to_lower().to_str_ascii(), ~"");
        assert_eq!("YMCA".to_ascii().to_lower().to_str_ascii(), ~"ymca");
        assert_eq!("abcDEFxyz:.;".to_ascii().to_upper().to_str_ascii(), ~"ABCDEFXYZ:.;");

        assert!("aBcDeF&?#".to_ascii().eq_ignore_case("AbCdEf&?#".to_ascii()));

        assert!("".is_ascii());
        assert!("a".is_ascii());
        assert!(!"\u2009".is_ascii());

    }

    #[test]
    fn test_owned_ascii_vec() {
        assert_eq!((~"( ;").into_ascii(), v2ascii!(~[40, 32, 59]));
        assert_eq!((~[40u8, 32u8, 59u8]).into_ascii(), v2ascii!(~[40, 32, 59]));
    }

    #[test]
    fn test_ascii_to_str() { assert_eq!(v2ascii!([40, 32, 59]).to_str_ascii(), ~"( ;"); }

    #[test]
    fn test_ascii_into_str() {
        assert_eq!(v2ascii!(~[40, 32, 59]).into_str(), ~"( ;");
    }

    #[test]
    fn test_ascii_to_bytes() {
        assert_eq!(v2ascii!(~[40, 32, 59]).into_bytes(), ~[40u8, 32u8, 59u8]);
    }

    #[test] #[should_fail]
    fn test_ascii_vec_fail_u8_slice()  { (&[127u8, 128u8, 255u8]).to_ascii(); }

    #[test] #[should_fail]
    fn test_ascii_vec_fail_str_slice() { "zoä华".to_ascii(); }

    #[test] #[should_fail]
    fn test_ascii_fail_u8_slice() { 255u8.to_ascii(); }

    #[test] #[should_fail]
    fn test_ascii_fail_char_slice() { 'λ'.to_ascii(); }

    #[test]
    fn test_to_ascii_upper() {
        assert_eq!("url()URL()uRl()ürl".to_ascii_upper(), ~"URL()URL()URL()üRL");
        assert_eq!("hıKß".to_ascii_upper(), ~"HıKß");

        let mut i = 0;
        while i <= 500 {
            let upper = if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 }
                        else { i };
            assert_eq!(from_char(from_u32(i).unwrap()).to_ascii_upper(),
                       from_char(from_u32(upper).unwrap()))
            i += 1;
        }
    }

    #[test]
    fn test_to_ascii_lower() {
        assert_eq!("url()URL()uRl()Ürl".to_ascii_lower(), ~"url()url()url()Ürl");
        // Dotted capital I, Kelvin sign, Sharp S.
        assert_eq!("HİKß".to_ascii_lower(), ~"hİKß");

        let mut i = 0;
        while i <= 500 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert_eq!(from_char(from_u32(i).unwrap()).to_ascii_lower(),
                       from_char(from_u32(lower).unwrap()))
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
            assert!(from_char(from_u32(i).unwrap()).
                eq_ignore_ascii_case(from_char(from_u32(lower).unwrap())));
            i += 1;
        }
    }

    #[test]
    fn test_to_str() {
        let s = Ascii{ chr: 't' as u8 }.to_str();
        assert_eq!(s, ~"t");
    }


}
