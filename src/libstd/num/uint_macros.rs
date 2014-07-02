// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![experimental]
#![macro_escape]
#![doc(hidden)]
#![allow(unsigned_negate)]

macro_rules! uint_module (($T:ty) => (

// String conversion functions and impl str -> num

/// Parse a byte slice as a number in the given base
///
/// Yields an `Option` because `buf` may or may not actually be parseable.
///
/// # Examples
///
/// ```
/// let num = std::uint::parse_bytes([49,50,51,52,53,54,55,56,57], 10);
/// assert!(num == Some(123456789));
/// ```
#[inline]
#[experimental = "might need to return Result"]
pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<$T> {
    strconv::from_str_bytes_common(buf, radix, false, false, false,
                                   strconv::ExpNone, false, false)
}

#[experimental = "might need to return Result"]
impl FromStr for $T {
    #[inline]
    fn from_str(s: &str) -> Option<$T> {
        strconv::from_str_common(s, 10u, false, false, false,
                                 strconv::ExpNone, false, false)
    }
}

#[experimental = "might need to return Result"]
impl FromStrRadix for $T {
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<$T> {
        strconv::from_str_common(s, radix, false, false, false,
                                 strconv::ExpNone, false, false)
    }
}

// String conversion functions and impl num -> str

/// Convert to a string as a byte slice in a given base.
///
/// Use in place of x.to_str() when you do not need to store the string permanently
///
/// # Examples
///
/// ```
/// #![allow(deprecated)]
///
/// std::uint::to_str_bytes(123, 10, |v| {
///     assert!(v == "123".as_bytes());
/// });
/// ```
#[inline]
#[deprecated = "just use .to_string(), or a BufWriter with write! if you mustn't allocate"]
pub fn to_str_bytes<U>(n: $T, radix: uint, f: |v: &[u8]| -> U) -> U {
    use io::{Writer, Seek};
    // The radix can be as low as 2, so we need at least 64 characters for a
    // base 2 number, and then we need another for a possible '-' character.
    let mut buf = [0u8, ..65];
    let amt = {
        let mut wr = ::io::BufWriter::new(buf);
        (write!(&mut wr, "{}", ::fmt::radix(n, radix as u8))).unwrap();
        wr.tell().unwrap() as uint
    };
    f(buf.slice(0, amt))
}

#[deprecated = "use fmt::radix"]
impl ToStrRadix for $T {
    /// Convert to a string in a given base.
    #[inline]
    fn to_str_radix(&self, radix: uint) -> String {
        format!("{}", ::fmt::radix(*self, radix as u8))
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    use num::ToStrRadix;
    use str::StrSlice;
    use u16;

    #[test]
    pub fn test_to_str() {
        assert_eq!((0 as $T).to_str_radix(10u), "0".to_string());
        assert_eq!((1 as $T).to_str_radix(10u), "1".to_string());
        assert_eq!((2 as $T).to_str_radix(10u), "2".to_string());
        assert_eq!((11 as $T).to_str_radix(10u), "11".to_string());
        assert_eq!((11 as $T).to_str_radix(16u), "b".to_string());
        assert_eq!((255 as $T).to_str_radix(16u), "ff".to_string());
        assert_eq!((0xff as $T).to_str_radix(10u), "255".to_string());
    }

    #[test]
    pub fn test_from_str() {
        assert_eq!(from_str::<$T>("0"), Some(0u as $T));
        assert_eq!(from_str::<$T>("3"), Some(3u as $T));
        assert_eq!(from_str::<$T>("10"), Some(10u as $T));
        assert_eq!(from_str::<u32>("123456789"), Some(123456789 as u32));
        assert_eq!(from_str::<$T>("00100"), Some(100u as $T));

        assert!(from_str::<$T>("").is_none());
        assert!(from_str::<$T>(" ").is_none());
        assert!(from_str::<$T>("x").is_none());
    }

    #[test]
    pub fn test_parse_bytes() {
        use str::StrSlice;
        assert_eq!(parse_bytes("123".as_bytes(), 10u), Some(123u as $T));
        assert_eq!(parse_bytes("1001".as_bytes(), 2u), Some(9u as $T));
        assert_eq!(parse_bytes("123".as_bytes(), 8u), Some(83u as $T));
        assert_eq!(u16::parse_bytes("123".as_bytes(), 16u), Some(291u as u16));
        assert_eq!(u16::parse_bytes("ffff".as_bytes(), 16u), Some(65535u as u16));
        assert_eq!(parse_bytes("z".as_bytes(), 36u), Some(35u as $T));

        assert!(parse_bytes("Z".as_bytes(), 10u).is_none());
        assert!(parse_bytes("_".as_bytes(), 2u).is_none());
    }

    #[test]
    fn test_uint_to_str_overflow() {
        let mut u8_val: u8 = 255_u8;
        assert_eq!(u8_val.to_str(), "255".to_string());

        u8_val += 1 as u8;
        assert_eq!(u8_val.to_str(), "0".to_string());

        let mut u16_val: u16 = 65_535_u16;
        assert_eq!(u16_val.to_str(), "65535".to_string());

        u16_val += 1 as u16;
        assert_eq!(u16_val.to_str(), "0".to_string());

        let mut u32_val: u32 = 4_294_967_295_u32;
        assert_eq!(u32_val.to_str(), "4294967295".to_string());

        u32_val += 1 as u32;
        assert_eq!(u32_val.to_str(), "0".to_string());

        let mut u64_val: u64 = 18_446_744_073_709_551_615_u64;
        assert_eq!(u64_val.to_str(), "18446744073709551615".to_string());

        u64_val += 1 as u64;
        assert_eq!(u64_val.to_str(), "0".to_string());
    }

    #[test]
    fn test_uint_from_str_overflow() {
        let mut u8_val: u8 = 255_u8;
        assert_eq!(from_str::<u8>("255"), Some(u8_val));
        assert!(from_str::<u8>("256").is_none());

        u8_val += 1 as u8;
        assert_eq!(from_str::<u8>("0"), Some(u8_val));
        assert!(from_str::<u8>("-1").is_none());

        let mut u16_val: u16 = 65_535_u16;
        assert_eq!(from_str::<u16>("65535"), Some(u16_val));
        assert!(from_str::<u16>("65536").is_none());

        u16_val += 1 as u16;
        assert_eq!(from_str::<u16>("0"), Some(u16_val));
        assert!(from_str::<u16>("-1").is_none());

        let mut u32_val: u32 = 4_294_967_295_u32;
        assert_eq!(from_str::<u32>("4294967295"), Some(u32_val));
        assert!(from_str::<u32>("4294967296").is_none());

        u32_val += 1 as u32;
        assert_eq!(from_str::<u32>("0"), Some(u32_val));
        assert!(from_str::<u32>("-1").is_none());

        let mut u64_val: u64 = 18_446_744_073_709_551_615_u64;
        assert_eq!(from_str::<u64>("18446744073709551615"), Some(u64_val));
        assert!(from_str::<u64>("18446744073709551616").is_none());

        u64_val += 1 as u64;
        assert_eq!(from_str::<u64>("0"), Some(u64_val));
        assert!(from_str::<u64>("-1").is_none());
    }

    #[test]
    #[should_fail]
    pub fn to_str_radix1() {
        100u.to_str_radix(1u);
    }

    #[test]
    #[should_fail]
    pub fn to_str_radix37() {
        100u.to_str_radix(37u);
    }
}

))
