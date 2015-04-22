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

macro_rules! uint_module { ($T:ident) => (

#[cfg(test)]
mod tests {
    use prelude::v1::*;

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
        assert_eq!($T::from_str_radix("123", 10), Ok(123 as $T));
        assert_eq!($T::from_str_radix("1001", 2), Ok(9 as $T));
        assert_eq!($T::from_str_radix("123", 8), Ok(83 as $T));
        assert_eq!(u16::from_str_radix("123", 16), Ok(291 as u16));
        assert_eq!(u16::from_str_radix("ffff", 16), Ok(65535 as u16));
        assert_eq!($T::from_str_radix("z", 36), Ok(35 as $T));

        assert_eq!($T::from_str_radix("Z", 10).ok(), None::<$T>);
        assert_eq!($T::from_str_radix("_", 2).ok(), None::<$T>);
    }
}

) }
