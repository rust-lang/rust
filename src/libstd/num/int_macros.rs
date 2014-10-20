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

macro_rules! int_module (($T:ty) => (

// String conversion functions and impl str -> num

/// Parse a byte slice as a number in the given base
///
/// Yields an `Option` because `buf` may or may not actually be parseable.
///
/// # Examples
///
/// ```
/// let num = std::i64::parse_bytes([49,50,51,52,53,54,55,56,57], 10);
/// assert!(num == Some(123456789));
/// ```
#[inline]
#[experimental = "might need to return Result"]
pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<$T> {
    strconv::from_str_bytes_common(buf, radix, true, false, false,
                               strconv::ExpNone, false, false)
}

#[experimental = "might need to return Result"]
impl FromStr for $T {
    #[inline]
    fn from_str(s: &str) -> Option<$T> {
        strconv::from_str_common(s, 10u, true, false, false,
                             strconv::ExpNone, false, false)
    }
}

#[experimental = "might need to return Result"]
impl FromStrRadix for $T {
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<$T> {
        strconv::from_str_common(s, radix, true, false, false,
                             strconv::ExpNone, false, false)
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    use i32;
    use str::StrSlice;

    #[test]
    fn test_from_str() {
        assert_eq!(from_str::<$T>("0"), Some(0 as $T));
        assert_eq!(from_str::<$T>("3"), Some(3 as $T));
        assert_eq!(from_str::<$T>("10"), Some(10 as $T));
        assert_eq!(from_str::<i32>("123456789"), Some(123456789 as i32));
        assert_eq!(from_str::<$T>("00100"), Some(100 as $T));

        assert_eq!(from_str::<$T>("-1"), Some(-1 as $T));
        assert_eq!(from_str::<$T>("-3"), Some(-3 as $T));
        assert_eq!(from_str::<$T>("-10"), Some(-10 as $T));
        assert_eq!(from_str::<i32>("-123456789"), Some(-123456789 as i32));
        assert_eq!(from_str::<$T>("-00100"), Some(-100 as $T));

        assert!(from_str::<$T>(" ").is_none());
        assert!(from_str::<$T>("x").is_none());
    }

    #[test]
    fn test_parse_bytes() {
        use str::StrSlice;
        assert_eq!(parse_bytes("123".as_bytes(), 10u), Some(123 as $T));
        assert_eq!(parse_bytes("1001".as_bytes(), 2u), Some(9 as $T));
        assert_eq!(parse_bytes("123".as_bytes(), 8u), Some(83 as $T));
        assert_eq!(i32::parse_bytes("123".as_bytes(), 16u), Some(291 as i32));
        assert_eq!(i32::parse_bytes("ffff".as_bytes(), 16u), Some(65535 as i32));
        assert_eq!(i32::parse_bytes("FFFF".as_bytes(), 16u), Some(65535 as i32));
        assert_eq!(parse_bytes("z".as_bytes(), 36u), Some(35 as $T));
        assert_eq!(parse_bytes("Z".as_bytes(), 36u), Some(35 as $T));

        assert_eq!(parse_bytes("-123".as_bytes(), 10u), Some(-123 as $T));
        assert_eq!(parse_bytes("-1001".as_bytes(), 2u), Some(-9 as $T));
        assert_eq!(parse_bytes("-123".as_bytes(), 8u), Some(-83 as $T));
        assert_eq!(i32::parse_bytes("-123".as_bytes(), 16u), Some(-291 as i32));
        assert_eq!(i32::parse_bytes("-ffff".as_bytes(), 16u), Some(-65535 as i32));
        assert_eq!(i32::parse_bytes("-FFFF".as_bytes(), 16u), Some(-65535 as i32));
        assert_eq!(parse_bytes("-z".as_bytes(), 36u), Some(-35 as $T));
        assert_eq!(parse_bytes("-Z".as_bytes(), 36u), Some(-35 as $T));

        assert!(parse_bytes("Z".as_bytes(), 35u).is_none());
        assert!(parse_bytes("-9".as_bytes(), 2u).is_none());
    }

    #[test]
    fn test_int_to_str_overflow() {
        let mut i8_val: i8 = 127_i8;
        assert_eq!(i8_val.to_string(), "127".to_string());

        i8_val += 1 as i8;
        assert_eq!(i8_val.to_string(), "-128".to_string());

        let mut i16_val: i16 = 32_767_i16;
        assert_eq!(i16_val.to_string(), "32767".to_string());

        i16_val += 1 as i16;
        assert_eq!(i16_val.to_string(), "-32768".to_string());

        let mut i32_val: i32 = 2_147_483_647_i32;
        assert_eq!(i32_val.to_string(), "2147483647".to_string());

        i32_val += 1 as i32;
        assert_eq!(i32_val.to_string(), "-2147483648".to_string());

        let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
        assert_eq!(i64_val.to_string(), "9223372036854775807".to_string());

        i64_val += 1 as i64;
        assert_eq!(i64_val.to_string(), "-9223372036854775808".to_string());
    }

    #[test]
    fn test_int_from_str_overflow() {
        let mut i8_val: i8 = 127_i8;
        assert_eq!(from_str::<i8>("127"), Some(i8_val));
        assert!(from_str::<i8>("128").is_none());

        i8_val += 1 as i8;
        assert_eq!(from_str::<i8>("-128"), Some(i8_val));
        assert!(from_str::<i8>("-129").is_none());

        let mut i16_val: i16 = 32_767_i16;
        assert_eq!(from_str::<i16>("32767"), Some(i16_val));
        assert!(from_str::<i16>("32768").is_none());

        i16_val += 1 as i16;
        assert_eq!(from_str::<i16>("-32768"), Some(i16_val));
        assert!(from_str::<i16>("-32769").is_none());

        let mut i32_val: i32 = 2_147_483_647_i32;
        assert_eq!(from_str::<i32>("2147483647"), Some(i32_val));
        assert!(from_str::<i32>("2147483648").is_none());

        i32_val += 1 as i32;
        assert_eq!(from_str::<i32>("-2147483648"), Some(i32_val));
        assert!(from_str::<i32>("-2147483649").is_none());

        let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
        assert_eq!(from_str::<i64>("9223372036854775807"), Some(i64_val));
        assert!(from_str::<i64>("9223372036854775808").is_none());

        i64_val += 1 as i64;
        assert_eq!(from_str::<i64>("-9223372036854775808"), Some(i64_val));
        assert!(from_str::<i64>("-9223372036854775809").is_none());
    }
}

))
