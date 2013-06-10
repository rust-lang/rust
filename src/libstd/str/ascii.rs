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
use cast;
use old_iter::BaseIter;
use iterator::IteratorUtil;
use vec::{CopyableVector, ImmutableVector, OwnedVector};

/// Datatype to hold one ascii character. It is 8 bit long.
#[deriving(Clone, Eq)]
pub struct Ascii { priv chr: u8 }

impl Ascii {
    /// Converts a ascii character into a `u8`.
    #[inline(always)]
    pub fn to_byte(self) -> u8 {
        self.chr
    }

    /// Converts a ascii character into a `char`.
    #[inline(always)]
    pub fn to_char(self) -> char {
        self.chr as char
    }

    /// Convert to lowercase.
    #[inline(always)]
    pub fn to_lower(self) -> Ascii {
        if self.chr >= 65 && self.chr <= 90 {
            Ascii{chr: self.chr | 0x20 }
        } else {
            self
        }
    }

    /// Convert to uppercase.
    #[inline(always)]
    pub fn to_upper(self) -> Ascii {
        if self.chr >= 97 && self.chr <= 122 {
            Ascii{chr: self.chr & !0x20 }
        } else {
            self
        }
    }

    /// Compares two ascii characters of equality, ignoring case.
    #[inline(always)]
    pub fn eq_ignore_case(self, other: Ascii) -> bool {
        self.to_lower().chr == other.to_lower().chr
    }
}

impl ToStr for Ascii {
    #[inline(always)]
    fn to_str(&self) -> ~str { str::from_bytes(['\'' as u8, self.chr, '\'' as u8]) }
}

/// Trait for converting into an ascii type.
pub trait AsciiCast<T> {
    /// Convert to an ascii type
    fn to_ascii(&self) -> T;

    /// Check if convertible to ascii
    fn is_ascii(&self) -> bool;
}

impl<'self> AsciiCast<&'self[Ascii]> for &'self [u8] {
    #[inline(always)]
    fn to_ascii(&self) -> &'self[Ascii] {
        assert!(self.is_ascii());
        unsafe{ cast::transmute(*self) }
    }

    #[inline(always)]
    fn is_ascii(&self) -> bool {
        for self.each |b| {
            if !b.is_ascii() { return false; }
        }
        true
    }
}

impl<'self> AsciiCast<&'self[Ascii]> for &'self str {
    #[inline(always)]
    fn to_ascii(&self) -> &'self[Ascii] {
        assert!(self.is_ascii());
        let (p,len): (*u8, uint) = unsafe{ cast::transmute(*self) };
        unsafe{ cast::transmute((p, len - 1))}
    }

    #[inline(always)]
    fn is_ascii(&self) -> bool {
        self.bytes_iter().all(|b| b.is_ascii())
    }
}

impl AsciiCast<Ascii> for u8 {
    #[inline(always)]
    fn to_ascii(&self) -> Ascii {
        assert!(self.is_ascii());
        Ascii{ chr: *self }
    }

    #[inline(always)]
    fn is_ascii(&self) -> bool {
        *self & 128 == 0u8
    }
}

impl AsciiCast<Ascii> for char {
    #[inline(always)]
    fn to_ascii(&self) -> Ascii {
        assert!(self.is_ascii());
        Ascii{ chr: *self as u8 }
    }

    #[inline(always)]
    fn is_ascii(&self) -> bool {
        *self - ('\x7F' & *self) == '\x00'
    }
}

/// Trait for copyless casting to an ascii vector.
pub trait OwnedAsciiCast {
    /// Take ownership and cast to an ascii vector without trailing zero element.
    fn to_ascii_consume(self) -> ~[Ascii];
}

impl OwnedAsciiCast for ~[u8] {
    #[inline(always)]
    fn to_ascii_consume(self) -> ~[Ascii] {
        assert!(self.is_ascii());
        unsafe {cast::transmute(self)}
    }
}

impl OwnedAsciiCast for ~str {
    #[inline(always)]
    fn to_ascii_consume(self) -> ~[Ascii] {
        assert!(self.is_ascii());
        let mut s = self;
        unsafe {
            str::raw::pop_byte(&mut s);
            cast::transmute(s)
        }
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

}

impl<'self> AsciiStr for &'self [Ascii] {
    #[inline(always)]
    fn to_str_ascii(&self) -> ~str {
        let mut cpy = self.to_owned();
        cpy.push(0u8.to_ascii());
        unsafe {cast::transmute(cpy)}
    }

    #[inline(always)]
    fn to_lower(&self) -> ~[Ascii] {
        self.map(|a| a.to_lower())
    }

    #[inline(always)]
    fn to_upper(&self) -> ~[Ascii] {
        self.map(|a| a.to_upper())
    }
}

impl ToStrConsume for ~[Ascii] {
    #[inline(always)]
    fn to_str_consume(self) -> ~str {
        let mut cpy = self;
        cpy.push(0u8.to_ascii());
        unsafe {cast::transmute(cpy)}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert!("".is_ascii());
        assert!("a".is_ascii());
        assert!(!"\u2009".is_ascii());

    }

    #[test]
    fn test_owned_ascii_vec() {
        // FIXME: #4318 Compiler crashes on moving self
        //assert_eq!(~"( ;".to_ascii_consume(), v2ascii!(~[40, 32, 59]));
        //assert_eq!(~[40u8, 32u8, 59u8].to_ascii_consume(), v2ascii!(~[40, 32, 59]));
        //assert_eq!(~"( ;".to_ascii_consume_with_null(), v2ascii!(~[40, 32, 59, 0]));
        //assert_eq!(~[40u8, 32u8, 59u8].to_ascii_consume_with_null(),
        //           v2ascii!(~[40, 32, 59, 0]));
    }

    #[test]
    fn test_ascii_to_str() { assert_eq!(v2ascii!([40, 32, 59]).to_str_ascii(), ~"( ;"); }

    #[test]
    fn test_ascii_to_str_consume() {
        // FIXME: #4318 Compiler crashes on moving self
        //assert_eq!(v2ascii!(~[40, 32, 59]).to_str_consume(), ~"( ;");
    }

    #[test] #[should_fail]
    fn test_ascii_vec_fail_u8_slice()  { (&[127u8, 128u8, 255u8]).to_ascii(); }

    #[test] #[should_fail]
    fn test_ascii_vec_fail_str_slice() { "zoä华".to_ascii(); }

    #[test] #[should_fail]
    fn test_ascii_fail_u8_slice() { 255u8.to_ascii(); }

    #[test] #[should_fail]
    fn test_ascii_fail_char_slice() { 'λ'.to_ascii(); }
}
