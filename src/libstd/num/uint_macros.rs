// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable]
#![doc(hidden)]
#![allow(unsigned_negation)]

macro_rules! uint_module { ($T:ty) => (

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use num::FromStrRadix;

    fn from_str<T: ::str::FromStr>(t: &str) -> Option<T> {
        ::str::FromStr::from_str(t)
    }

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
}

) }
