// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Additional functionality for numerics.
//!
//! This module provides some extra types that are useful when doing numerical
//! work. See the individual documentation for each piece for more information.

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
pub use core::num::{Zero, One};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::num::{FpCategory, ParseIntError, ParseFloatError, TryFromIntError};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::num::Wrapping;

#[cfg(test)] use fmt;
#[cfg(test)] use ops::{Add, Sub, Mul, Div, Rem};

/// Helper function for testing numeric operations
#[cfg(test)]
pub fn test_num<T>(ten: T, two: T) where
    T: PartialEq
     + Add<Output=T> + Sub<Output=T>
     + Mul<Output=T> + Div<Output=T>
     + Rem<Output=T> + fmt::Debug
     + Copy
{
    assert_eq!(ten.add(two),  ten + two);
    assert_eq!(ten.sub(two),  ten - two);
    assert_eq!(ten.mul(two),  ten * two);
    assert_eq!(ten.div(two),  ten / two);
    assert_eq!(ten.rem(two),  ten % two);
}

#[cfg(test)]
mod tests {
    use u8;
    use u16;
    use u32;
    use u64;
    use usize;
    use ops::Mul;

    #[test]
    fn test_saturating_add_uint() {
        use usize::MAX;
        assert_eq!(3_usize.saturating_add(5_usize), 8_usize);
        assert_eq!(3_usize.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
    }

    #[test]
    fn test_saturating_sub_uint() {
        use usize::MAX;
        assert_eq!(5_usize.saturating_sub(3_usize), 2_usize);
        assert_eq!(3_usize.saturating_sub(5_usize), 0_usize);
        assert_eq!(0_usize.saturating_sub(1_usize), 0_usize);
        assert_eq!((MAX-1).saturating_sub(MAX), 0);
    }

    #[test]
    fn test_saturating_add_int() {
        use isize::{MIN,MAX};
        assert_eq!(3i32.saturating_add(5), 8);
        assert_eq!(3isize.saturating_add(MAX-1), MAX);
        assert_eq!(MAX.saturating_add(MAX), MAX);
        assert_eq!((MAX-2).saturating_add(1), MAX-1);
        assert_eq!(3i32.saturating_add(-5), -2);
        assert_eq!(MIN.saturating_add(-1), MIN);
        assert_eq!((-2isize).saturating_add(-MAX), MIN);
    }

    #[test]
    fn test_saturating_sub_int() {
        use isize::{MIN,MAX};
        assert_eq!(3i32.saturating_sub(5), -2);
        assert_eq!(MIN.saturating_sub(1), MIN);
        assert_eq!((-2isize).saturating_sub(MAX), MIN);
        assert_eq!(3i32.saturating_sub(-5), 8);
        assert_eq!(3isize.saturating_sub(-(MAX-1)), MAX);
        assert_eq!(MAX.saturating_sub(-MAX), MAX);
        assert_eq!((MAX-2).saturating_sub(-1), MAX-1);
    }

    #[test]
    fn test_checked_add() {
        let five_less = usize::MAX - 5;
        assert_eq!(five_less.checked_add(0), Some(usize::MAX - 5));
        assert_eq!(five_less.checked_add(1), Some(usize::MAX - 4));
        assert_eq!(five_less.checked_add(2), Some(usize::MAX - 3));
        assert_eq!(five_less.checked_add(3), Some(usize::MAX - 2));
        assert_eq!(five_less.checked_add(4), Some(usize::MAX - 1));
        assert_eq!(five_less.checked_add(5), Some(usize::MAX));
        assert_eq!(five_less.checked_add(6), None);
        assert_eq!(five_less.checked_add(7), None);
    }

    #[test]
    fn test_checked_sub() {
        assert_eq!(5_usize.checked_sub(0), Some(5));
        assert_eq!(5_usize.checked_sub(1), Some(4));
        assert_eq!(5_usize.checked_sub(2), Some(3));
        assert_eq!(5_usize.checked_sub(3), Some(2));
        assert_eq!(5_usize.checked_sub(4), Some(1));
        assert_eq!(5_usize.checked_sub(5), Some(0));
        assert_eq!(5_usize.checked_sub(6), None);
        assert_eq!(5_usize.checked_sub(7), None);
    }

    #[test]
    fn test_checked_mul() {
        let third = usize::MAX / 3;
        assert_eq!(third.checked_mul(0), Some(0));
        assert_eq!(third.checked_mul(1), Some(third));
        assert_eq!(third.checked_mul(2), Some(third * 2));
        assert_eq!(third.checked_mul(3), Some(third * 3));
        assert_eq!(third.checked_mul(4), None);
    }

    macro_rules! test_is_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).is_power_of_two(), false);
                assert_eq!((1 as $T).is_power_of_two(), true);
                assert_eq!((2 as $T).is_power_of_two(), true);
                assert_eq!((3 as $T).is_power_of_two(), false);
                assert_eq!((4 as $T).is_power_of_two(), true);
                assert_eq!((5 as $T).is_power_of_two(), false);
                assert_eq!(($T::MAX / 2 + 1).is_power_of_two(), true);
            }
        )
    }

    test_is_power_of_two!{ test_is_power_of_two_u8, u8 }
    test_is_power_of_two!{ test_is_power_of_two_u16, u16 }
    test_is_power_of_two!{ test_is_power_of_two_u32, u32 }
    test_is_power_of_two!{ test_is_power_of_two_u64, u64 }
    test_is_power_of_two!{ test_is_power_of_two_uint, usize }

    macro_rules! test_next_power_of_two {
        ($test_name:ident, $T:ident) => (
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).next_power_of_two(), 1);
                let mut next_power = 1;
                for i in 1 as $T..40 {
                     assert_eq!(i.next_power_of_two(), next_power);
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    }

    test_next_power_of_two! { test_next_power_of_two_u8, u8 }
    test_next_power_of_two! { test_next_power_of_two_u16, u16 }
    test_next_power_of_two! { test_next_power_of_two_u32, u32 }
    test_next_power_of_two! { test_next_power_of_two_u64, u64 }
    test_next_power_of_two! { test_next_power_of_two_uint, usize }

    macro_rules! test_checked_next_power_of_two {
        ($test_name:ident, $T:ident) => (
            #[cfg_attr(target_os = "emscripten", ignore)] // FIXME(#39119)
            fn $test_name() {
                #![test]
                assert_eq!((0 as $T).checked_next_power_of_two(), Some(1));
                assert!(($T::MAX / 2).checked_next_power_of_two().is_some());
                assert_eq!(($T::MAX - 1).checked_next_power_of_two(), None);
                assert_eq!($T::MAX.checked_next_power_of_two(), None);
                let mut next_power = 1;
                for i in 1 as $T..40 {
                     assert_eq!(i.checked_next_power_of_two(), Some(next_power));
                     if i == next_power { next_power *= 2 }
                }
            }
        )
    }

    test_checked_next_power_of_two! { test_checked_next_power_of_two_u8, u8 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u16, u16 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u32, u32 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_u64, u64 }
    test_checked_next_power_of_two! { test_checked_next_power_of_two_uint, usize }

    #[test]
    fn test_pow() {
        fn naive_pow<T: Mul<Output=T> + Copy>(one: T, base: T, exp: usize) -> T {
            (0..exp).fold(one, |acc, _| acc * base)
        }
        macro_rules! assert_pow {
            (($num:expr, $exp:expr) => $expected:expr) => {{
                let result = $num.pow($exp);
                assert_eq!(result, $expected);
                assert_eq!(result, naive_pow(1, $num, $exp));
            }}
        }
        assert_pow!((3u32,     0 ) => 1);
        assert_pow!((5u32,     1 ) => 5);
        assert_pow!((-4i32,    2 ) => 16);
        assert_pow!((8u32,     3 ) => 512);
        assert_pow!((2u64,     50) => 1125899906842624);
    }

    #[test]
    fn test_uint_to_str_overflow() {
        let mut u8_val: u8 = 255;
        assert_eq!(u8_val.to_string(), "255");

        u8_val = u8_val.wrapping_add(1);
        assert_eq!(u8_val.to_string(), "0");

        let mut u16_val: u16 = 65_535;
        assert_eq!(u16_val.to_string(), "65535");

        u16_val = u16_val.wrapping_add(1);
        assert_eq!(u16_val.to_string(), "0");

        let mut u32_val: u32 = 4_294_967_295;
        assert_eq!(u32_val.to_string(), "4294967295");

        u32_val = u32_val.wrapping_add(1);
        assert_eq!(u32_val.to_string(), "0");

        let mut u64_val: u64 = 18_446_744_073_709_551_615;
        assert_eq!(u64_val.to_string(), "18446744073709551615");

        u64_val = u64_val.wrapping_add(1);
        assert_eq!(u64_val.to_string(), "0");
    }

    fn from_str<T: ::str::FromStr>(t: &str) -> Option<T> {
        ::str::FromStr::from_str(t).ok()
    }

    #[test]
    fn test_uint_from_str_overflow() {
        let mut u8_val: u8 = 255;
        assert_eq!(from_str::<u8>("255"), Some(u8_val));
        assert_eq!(from_str::<u8>("256"), None);

        u8_val = u8_val.wrapping_add(1);
        assert_eq!(from_str::<u8>("0"), Some(u8_val));
        assert_eq!(from_str::<u8>("-1"), None);

        let mut u16_val: u16 = 65_535;
        assert_eq!(from_str::<u16>("65535"), Some(u16_val));
        assert_eq!(from_str::<u16>("65536"), None);

        u16_val = u16_val.wrapping_add(1);
        assert_eq!(from_str::<u16>("0"), Some(u16_val));
        assert_eq!(from_str::<u16>("-1"), None);

        let mut u32_val: u32 = 4_294_967_295;
        assert_eq!(from_str::<u32>("4294967295"), Some(u32_val));
        assert_eq!(from_str::<u32>("4294967296"), None);

        u32_val = u32_val.wrapping_add(1);
        assert_eq!(from_str::<u32>("0"), Some(u32_val));
        assert_eq!(from_str::<u32>("-1"), None);

        let mut u64_val: u64 = 18_446_744_073_709_551_615;
        assert_eq!(from_str::<u64>("18446744073709551615"), Some(u64_val));
        assert_eq!(from_str::<u64>("18446744073709551616"), None);

        u64_val = u64_val.wrapping_add(1);
        assert_eq!(from_str::<u64>("0"), Some(u64_val));
        assert_eq!(from_str::<u64>("-1"), None);
    }
}


#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;

    #[bench]
    fn bench_pow_function(b: &mut Bencher) {
        let v = (0..1024).collect::<Vec<u32>>();
        b.iter(|| {v.iter().fold(0u32, |old, new| old.pow(*new as u32));});
    }
}
