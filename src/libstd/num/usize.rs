// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for pointer-sized unsigned integers (`usize` type)
//!
//! This type was recently added to replace `uint`. The rollout of the
//! new type will gradually take place over the alpha cycle along with
//! the development of clearer conventions around integer types.

#![stable(feature = "rust1", since = "1.0.0")]
#![doc(primitive = "usize")]

pub use core::usize::{BITS, BYTES, MIN, MAX};

uint_module! { usize }

#[cfg(test)]
mod tests {
    use prelude::v1::*;

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
