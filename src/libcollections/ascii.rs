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

use Collection;
use core::iter::Iterator;
use core::mem;
use core::option::{Option, Some, None};
use core::slice::{Vector, ImmutableVector};
use std::hash::{Hash, Writer};
use str::{Str, StrSlice};
use str;
use string::String;
use vec::Vec;

pub use core::ascii::{Ascii, AsciiCast};
pub use core::ascii::{ASCII_LOWER_MAP, ASCII_UPPER_MAP};

#[deprecated="this trait has been renamed to `AsciiExt`"]
pub use StrAsciiExt = self::AsciiExt;

#[deprecated="this trait has been renamed to `OwnedAsciiExt`"]
pub use OwnedStrAsciiExt = self::OwnedAsciiExt;

#[cfg(not(test))]
impl<S: Writer> Hash<S> for Ascii {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.to_byte().hash(state);
    }
}

/// Trait for copyless casting to an ascii vector.
pub trait OwnedAsciiCast {
    /// Check if convertible to ascii
    fn is_ascii(&self) -> bool;

    /// Take ownership and cast to an ascii vector. Fail on non-ASCII input.
    #[inline]
    fn into_ascii(self) -> Vec<Ascii> {
        assert!(self.is_ascii());
        unsafe {self.into_ascii_nocheck()}
    }

    /// Take ownership and cast to an ascii vector. Return None on non-ASCII input.
    #[inline]
    fn into_ascii_opt(self) -> Option<Vec<Ascii>> {
        if self.is_ascii() {
            Some(unsafe { self.into_ascii_nocheck() })
        } else {
            None
        }
    }

    /// Take ownership and cast to an ascii vector.
    /// Does not perform validation checks.
    unsafe fn into_ascii_nocheck(self) -> Vec<Ascii>;
}

impl OwnedAsciiCast for String {
    #[inline]
    fn is_ascii(&self) -> bool {
        self.as_slice().is_ascii()
    }

    #[inline]
    unsafe fn into_ascii_nocheck(self) -> Vec<Ascii> {
        let v: Vec<u8> = mem::transmute(self);
        v.into_ascii_nocheck()
    }
}

impl OwnedAsciiCast for Vec<u8> {
    #[inline]
    fn is_ascii(&self) -> bool {
        self.as_slice().is_ascii()
    }

    #[inline]
    unsafe fn into_ascii_nocheck(self) -> Vec<Ascii> {
        mem::transmute(self)
    }
}

/// Trait for converting an ascii type to a string. Needed to convert
/// `&[Ascii]` to `&str`.
pub trait AsciiStr {
    /// Convert to a string.
    fn as_str_ascii<'a>(&'a self) -> &'a str;

    /// Convert to vector representing a lower cased ascii string.
    fn to_lower(&self) -> Vec<Ascii>;

    /// Convert to vector representing a upper cased ascii string.
    fn to_upper(&self) -> Vec<Ascii>;

    /// Compares two Ascii strings ignoring case.
    fn eq_ignore_case(self, other: &[Ascii]) -> bool;
}

impl<'a> AsciiStr for &'a [Ascii] {
    #[inline]
    fn as_str_ascii<'a>(&'a self) -> &'a str {
        unsafe { mem::transmute(*self) }
    }

    #[inline]
    fn to_lower(&self) -> Vec<Ascii> {
        self.iter().map(|a| a.to_lowercase()).collect()
    }

    #[inline]
    fn to_upper(&self) -> Vec<Ascii> {
        self.iter().map(|a| a.to_uppercase()).collect()
    }

    #[inline]
    fn eq_ignore_case(self, other: &[Ascii]) -> bool {
        self.iter().zip(other.iter()).all(|(&a, &b)| a.eq_ignore_case(b))
    }
}

/// Trait to convert to an owned byte vector by consuming self
pub trait IntoBytes {
    /// Converts to an owned byte vector by consuming self
    fn into_bytes(self) -> Vec<u8>;
}

impl IntoBytes for Vec<Ascii> {
    fn into_bytes(self) -> Vec<u8> {
        unsafe { mem::transmute(self) }
    }
}

/// Extension methods for ASCII-subset only operations on owned strings
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

impl OwnedAsciiExt for String {
    #[inline]
    fn into_ascii_upper(self) -> String {
        // Vec<u8>::into_ascii_upper() preserves the UTF-8 invariant.
        unsafe { str::raw::from_utf8_owned(self.into_bytes().into_ascii_upper()) }
    }

    #[inline]
    fn into_ascii_lower(self) -> String {
        // Vec<u8>::into_ascii_lower() preserves the UTF-8 invariant.
        unsafe { str::raw::from_utf8_owned(self.into_bytes().into_ascii_lower()) }
    }
}

impl OwnedAsciiExt for Vec<u8> {
    #[inline]
    fn into_ascii_upper(mut self) -> Vec<u8> {
        for byte in self.mut_iter() {
            *byte = ASCII_UPPER_MAP[*byte as uint];
        }
        self
    }

    #[inline]
    fn into_ascii_lower(mut self) -> Vec<u8> {
        for byte in self.mut_iter() {
            *byte = ASCII_LOWER_MAP[*byte as uint];
        }
        self
    }
}

/// Extension methods for ASCII-subset only operations on string slices
pub trait AsciiExt<T> {
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
    fn eq_ignore_ascii_case(&self, other: Self) -> bool;
}

impl<'a> AsciiExt<String> for &'a str {
    #[inline]
    fn to_ascii_upper(&self) -> String {
        // Vec<u8>::to_ascii_upper() preserves the UTF-8 invariant.
        unsafe { str::raw::from_utf8_owned(self.as_bytes().to_ascii_upper()) }
    }

    #[inline]
    fn to_ascii_lower(&self) -> String {
        // Vec<u8>::to_ascii_lower() preserves the UTF-8 invariant.
        unsafe { str::raw::from_utf8_owned(self.as_bytes().to_ascii_lower()) }
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    }
}

impl<'a> AsciiExt<Vec<u8>> for &'a [u8] {
    #[inline]
    fn to_ascii_upper(&self) -> Vec<u8> {
        self.iter().map(|&byte| ASCII_UPPER_MAP[byte as uint]).collect()
    }

    #[inline]
    fn to_ascii_lower(&self) -> Vec<u8> {
        self.iter().map(|&byte| ASCII_LOWER_MAP[byte as uint]).collect()
    }

    #[inline]
    fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(
            |(byte_self, byte_other)| {
                ASCII_LOWER_MAP[*byte_self as uint] ==
                    ASCII_LOWER_MAP[*byte_other as uint]
            })
    }
}

#[cfg(test)]
mod tests {
    use MutableSeq;
    use core::char::from_u32;
    use core::prelude::*;
    use str::StrSlice;
    use string::String;
    use super::*;
    use vec::Vec;
    use core::mem;

    macro_rules! ascii (
        ($e:expr) => (unsafe {mem::transmute::<u8, Ascii>($e)});
    )

    macro_rules! v2ascii (
        ( [$($e:expr),*]) => (&[$(ascii!{$e}),*]);
        (&[$($e:expr),*]) => (&[$(ascii!{$e}),*]);
    )

    macro_rules! vec2ascii (
        ($($e:expr),*) => (Vec::from_slice([$(ascii!{$e}),*]));
    )

    trait StdBypass {
        fn to_string(&self) -> String;
        fn into_string(self) -> String { self.to_string() }
    }

    impl StdBypass for Vec<Ascii> {
        fn to_string(&self) -> String {
            unsafe { mem::transmute(self.clone()) }
        }
    }

    impl<'a> StdBypass for &'a str {
        fn to_string(&self) -> String {
            String::from_str(*self)
        }
    }

    impl<'a> StdBypass for Ascii {
        fn to_string(&self) -> String {
            String::from_byte(self.to_byte())
        }
    }

    impl<'a> StdBypass for char {
        fn to_string(&self) -> String {
            String::from_char(1, *self)
        }
    }

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

        assert!('0'.to_ascii().is_digit());
        assert!('9'.to_ascii().is_digit());
        assert!(!'/'.to_ascii().is_digit());
        assert!(!':'.to_ascii().is_digit());

        assert!((0x1fu8).to_ascii().is_control());
        assert!(!' '.to_ascii().is_control());
        assert!((0x7fu8).to_ascii().is_control());

        assert!("banana".chars().all(|c| c.is_ascii()));
        assert!(!"ประเทศไทย中华Việt Nam".chars().all(|c| c.is_ascii()));
    }

    #[test]
    fn test_ascii_vec() {
        let test = &[40u8, 32u8, 59u8];
        assert_eq!(test.to_ascii(), v2ascii!([40, 32, 59]));
        assert_eq!("( ;".to_ascii(), v2ascii!([40, 32, 59]));
        let v = vec![40u8, 32u8, 59u8];
        assert_eq!(v.as_slice().to_ascii(), v2ascii!([40, 32, 59]));
        let s = "( ;".to_string();
        assert_eq!(s.as_slice().to_ascii(), v2ascii!([40, 32, 59]));

        assert_eq!("abCDef&?#".to_ascii().to_lower().into_string(), "abcdef&?#".to_string());
        assert_eq!("abCDef&?#".to_ascii().to_upper().into_string(), "ABCDEF&?#".to_string());

        assert_eq!("".to_ascii().to_lower().into_string(), "".to_string());
        assert_eq!("YMCA".to_ascii().to_lower().into_string(), "ymca".to_string());
        assert_eq!("abcDEFxyz:.;".to_ascii().to_upper().into_string(), "ABCDEFXYZ:.;".to_string());

        assert!("aBcDeF&?#".to_ascii().eq_ignore_case("AbCdEf&?#".to_ascii()));

        assert!("".is_ascii());
        assert!("a".is_ascii());
        assert!(!"\u2009".is_ascii());

    }

    #[test]
    fn test_ascii_vec_ng() {
        assert_eq!("abCDef&?#".to_ascii().to_lower().into_string(), "abcdef&?#".to_string());
        assert_eq!("abCDef&?#".to_ascii().to_upper().into_string(), "ABCDEF&?#".to_string());
        assert_eq!("".to_ascii().to_lower().into_string(), "".to_string());
        assert_eq!("YMCA".to_ascii().to_lower().into_string(), "ymca".to_string());
        assert_eq!("abcDEFxyz:.;".to_ascii().to_upper().into_string(), "ABCDEFXYZ:.;".to_string());
    }

    #[test]
    fn test_owned_ascii_vec() {
        assert_eq!(("( ;".to_string()).into_ascii(), vec2ascii![40, 32, 59]);
        assert_eq!((vec![40u8, 32u8, 59u8]).into_ascii(), vec2ascii![40, 32, 59]);
    }

    #[test]
    fn test_ascii_as_str() {
        let v = v2ascii!([40, 32, 59]);
        assert_eq!(v.as_str_ascii(), "( ;");
    }

    #[test]
    fn test_ascii_into_string() {
        assert_eq!(vec2ascii![40, 32, 59].into_string(), "( ;".to_string());
        assert_eq!(vec2ascii!(40, 32, 59).into_string(), "( ;".to_string());
    }

    #[test]
    fn test_ascii_to_bytes() {
        assert_eq!(vec2ascii![40, 32, 59].into_bytes(), vec![40u8, 32u8, 59u8]);
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
    fn test_opt() {
        assert_eq!(65u8.to_ascii_opt(), Some(ascii!{ 65u8 }));
        assert_eq!(255u8.to_ascii_opt(), None);

        assert_eq!('A'.to_ascii_opt(), Some(ascii! { 65u8 }));
        assert_eq!('λ'.to_ascii_opt(), None);

        assert_eq!("zoä华".to_ascii_opt(), None);

        let test1 = &[127u8, 128u8, 255u8];
        assert_eq!((test1).to_ascii_opt(), None);

        let v = [40u8, 32u8, 59u8];
        let v2 = v2ascii!(&[40, 32, 59]);
        assert_eq!(v.to_ascii_opt(), Some(v2));
        let v = [127u8, 128u8, 255u8];
        assert_eq!(v.to_ascii_opt(), None);

        let v = "( ;";
        let v2 = v2ascii!(&[40, 32, 59]);
        assert_eq!(v.to_ascii_opt(), Some(v2));
        assert_eq!("zoä华".to_ascii_opt(), None);

        assert_eq!((vec![40u8, 32u8, 59u8]).into_ascii_opt(), Some(vec2ascii![40, 32, 59]));
        assert_eq!((vec![127u8, 128u8, 255u8]).into_ascii_opt(), None);

        assert_eq!(("( ;".to_string()).into_ascii_opt(), Some(vec2ascii![40, 32, 59]));
        assert_eq!(("zoä华".to_string()).into_ascii_opt(), None);
    }

    #[test]
    fn test_to_ascii_upper() {
        assert_eq!("url()URL()uRl()ürl".to_ascii_upper(), "URL()URL()URL()üRL".to_string());
        assert_eq!("hıKß".to_ascii_upper(), "HıKß".to_string());

        let mut i = 0;
        while i <= 500 {
            let upper = if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().as_slice().to_ascii_upper(),
                       (from_u32(upper).unwrap()).to_string())
            i += 1;
        }
    }

    #[test]
    fn test_to_ascii_lower() {
        assert_eq!("url()URL()uRl()Ürl".to_ascii_lower(), "url()url()url()Ürl".to_string());
        // Dotted capital I, Kelvin sign, Sharp S.
        assert_eq!("HİKß".to_ascii_lower(), "hİKß".to_string());

        let mut i = 0;
        while i <= 500 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().as_slice().to_ascii_lower(),
                       (from_u32(lower).unwrap()).to_string())
            i += 1;
        }
    }

    #[test]
    fn test_into_ascii_upper() {
        assert_eq!(("url()URL()uRl()ürl".to_string()).into_ascii_upper(),
                   "URL()URL()URL()üRL".to_string());
        assert_eq!(("hıKß".to_string()).into_ascii_upper(), "HıKß".to_string());

        let mut i = 0;
        while i <= 500 {
            let upper = if 'a' as u32 <= i && i <= 'z' as u32 { i + 'A' as u32 - 'a' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().into_ascii_upper(),
                       (from_u32(upper).unwrap()).to_string())
            i += 1;
        }
    }

    #[test]
    fn test_into_ascii_lower() {
        assert_eq!(("url()URL()uRl()Ürl".to_string()).into_ascii_lower(),
                   "url()url()url()Ürl".to_string());
        // Dotted capital I, Kelvin sign, Sharp S.
        assert_eq!(("HİKß".to_string()).into_ascii_lower(), "hİKß".to_string());

        let mut i = 0;
        while i <= 500 {
            let lower = if 'A' as u32 <= i && i <= 'Z' as u32 { i + 'a' as u32 - 'A' as u32 }
                        else { i };
            assert_eq!((from_u32(i).unwrap()).to_string().into_ascii_lower(),
                       (from_u32(lower).unwrap()).to_string())
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
            assert!((from_u32(i).unwrap()).to_string().as_slice().eq_ignore_ascii_case(
                    (from_u32(lower).unwrap()).to_string().as_slice()));
            i += 1;
        }
    }
}
