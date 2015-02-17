// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "std_misc")]
#![doc(hidden)]
#![allow(unsigned_negation)]

macro_rules! uint_module { ($T:ty) => (

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use num::FromStrRadix;

    fn from_str<T: ::str::FromStr>(t: &str) -> Option<T> {
        ::str::FromStr::from_str(t).ok()
    }

    #[test]
    pub fn test_from_str() {
        assert_eq!(from_str::<$T>("0"), Some(0 as $T));
        assert_eq!(from_str::<$T>("3"), Some(3 as $T));
        assert_eq!(from_str::<$T>("10"), Some(10 as $T));
        assert_eq!(from_str::<u32>("123456789"), Some(123456789 as u32));
        assert_eq!(from_str::<$T>("00100"), Some(100 as $T));

        assert_eq!(from_str::<$T>(""), None);
        assert_eq!(from_str::<$T>(" "), None);
        assert_eq!(from_str::<$T>("x"), None);
    }

    #[test]
    pub fn test_parse_bytes() {
        assert_eq!(FromStrRadix::from_str_radix("123", 10), Ok(123 as $T));
        assert_eq!(FromStrRadix::from_str_radix("1001", 2), Ok(9 as $T));
        assert_eq!(FromStrRadix::from_str_radix("123", 8), Ok(83 as $T));
        assert_eq!(FromStrRadix::from_str_radix("123", 16), Ok(291 as u16));
        assert_eq!(FromStrRadix::from_str_radix("ffff", 16), Ok(65535 as u16));
        assert_eq!(FromStrRadix::from_str_radix("z", 36), Ok(35 as $T));

        assert_eq!(FromStrRadix::from_str_radix("Z", 10).ok(), None::<$T>);
        assert_eq!(FromStrRadix::from_str_radix("_", 2).ok(), None::<$T>);
    }

    #[test]
    fn test_uint_to_str_overflow() {
        let mut u8_val: u8 = 255_u8;
        assert_eq!(u8_val.to_string(), "255");

        u8_val += 1 as u8;
        assert_eq!(u8_val.to_string(), "0");

        let mut u16_val: u16 = 65_535_u16;
        assert_eq!(u16_val.to_string(), "65535");

        u16_val += 1 as u16;
        assert_eq!(u16_val.to_string(), "0");

        let mut u32_val: u32 = 4_294_967_295_u32;
        assert_eq!(u32_val.to_string(), "4294967295");

        u32_val += 1 as u32;
        assert_eq!(u32_val.to_string(), "0");

        let mut u64_val: u64 = 18_446_744_073_709_551_615_u64;
        assert_eq!(u64_val.to_string(), "18446744073709551615");

        u64_val += 1 as u64;
        assert_eq!(u64_val.to_string(), "0");
    }

    #[test]
    fn test_uint_from_str_overflow() {
        let mut u8_val: u8 = 255_u8;
        assert_eq!(from_str::<u8>("255"), Some(u8_val));
        assert_eq!(from_str::<u8>("256"), None);

        u8_val += 1 as u8;
        assert_eq!(from_str::<u8>("0"), Some(u8_val));
        assert_eq!(from_str::<u8>("-1"), None);

        let mut u16_val: u16 = 65_535_u16;
        assert_eq!(from_str::<u16>("65535"), Some(u16_val));
        assert_eq!(from_str::<u16>("65536"), None);

        u16_val += 1 as u16;
        assert_eq!(from_str::<u16>("0"), Some(u16_val));
        assert_eq!(from_str::<u16>("-1"), None);

        let mut u32_val: u32 = 4_294_967_295_u32;
        assert_eq!(from_str::<u32>("4294967295"), Some(u32_val));
        assert_eq!(from_str::<u32>("4294967296"), None);

        u32_val += 1 as u32;
        assert_eq!(from_str::<u32>("0"), Some(u32_val));
        assert_eq!(from_str::<u32>("-1"), None);

        let mut u64_val: u64 = 18_446_744_073_709_551_615_u64;
        assert_eq!(from_str::<u64>("18446744073709551615"), Some(u64_val));
        assert_eq!(from_str::<u64>("18446744073709551616"), None);

        u64_val += 1 as u64;
        assert_eq!(from_str::<u64>("0"), Some(u64_val));
        assert_eq!(from_str::<u64>("-1"), None);
    }
}

) }
