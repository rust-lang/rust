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
#![allow(unsigned_negation)]

macro_rules! uint_module (($T:ty) => (

// String conversion functions and impl num -> str

/// Convert to a string as a byte slice in a given base.
///
/// Use in place of x.to_string() when you do not need to store the string permanently
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
        let mut wr = ::io::BufWriter::new(&mut buf);
        (write!(&mut wr, "{}", ::fmt::radix(n, radix as u8))).unwrap();
        wr.tell().unwrap() as uint
    };
    f(buf[..amt])
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use num::FromStrRadix;

    #[test]
    pub fn test_from_str() {
        assert_eq!(from_str::<$T>("0"), Some(0u as $T));
        assert_eq!(from_str::<$T>("3"), Some(3u as $T));
        assert_eq!(from_str::<$T>("10"), Some(10u as $T));
        assert_eq!(from_str::<u32>("123456789"), Some(123456789 as u32));
        assert_eq!(from_str::<$T>("00100"), Some(100u as $T));

        assert_eq!(from_str::<$T>(""), None);
        assert_eq!(from_str::<$T>(" "), None);
        assert_eq!(from_str::<$T>("x"), None);
    }

    #[test]
    pub fn test_parse_bytes() {
        assert_eq!(FromStrRadix::from_str_radix("123", 10), Some(123u as $T));
        assert_eq!(FromStrRadix::from_str_radix("1001", 2), Some(9u as $T));
        assert_eq!(FromStrRadix::from_str_radix("123", 8), Some(83u as $T));
        assert_eq!(FromStrRadix::from_str_radix("123", 16), Some(291u as u16));
        assert_eq!(FromStrRadix::from_str_radix("ffff", 16), Some(65535u as u16));
        assert_eq!(FromStrRadix::from_str_radix("z", 36), Some(35u as $T));

        assert_eq!(FromStrRadix::from_str_radix("Z", 10), None::<$T>);
        assert_eq!(FromStrRadix::from_str_radix("_", 2), None::<$T>);
    }

    #[test]
    fn test_uint_to_str_overflow() {
        let mut u8_val: u8 = 255_u8;
        assert_eq!(u8_val.to_string(), "255".to_string());

        u8_val += 1 as u8;
        assert_eq!(u8_val.to_string(), "0".to_string());

        let mut u16_val: u16 = 65_535_u16;
        assert_eq!(u16_val.to_string(), "65535".to_string());

        u16_val += 1 as u16;
        assert_eq!(u16_val.to_string(), "0".to_string());

        let mut u32_val: u32 = 4_294_967_295_u32;
        assert_eq!(u32_val.to_string(), "4294967295".to_string());

        u32_val += 1 as u32;
        assert_eq!(u32_val.to_string(), "0".to_string());

        let mut u64_val: u64 = 18_446_744_073_709_551_615_u64;
        assert_eq!(u64_val.to_string(), "18446744073709551615".to_string());

        u64_val += 1 as u64;
        assert_eq!(u64_val.to_string(), "0".to_string());
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

))
