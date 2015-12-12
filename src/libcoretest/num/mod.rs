// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::PartialEq;
use core::fmt::Debug;
use core::ops::{Add, Sub, Mul, Div, Rem};
use core::marker::Copy;

#[macro_use]
mod int_macros;

mod i8;
mod i16;
mod i32;
mod i64;

#[macro_use]
mod uint_macros;

mod u8;
mod u16;
mod u32;
mod u64;

mod flt2dec;
mod dec2flt;
mod bignum;

/// Helper function for testing numeric operations
pub fn test_num<T>(ten: T, two: T) where
    T: PartialEq
     + Add<Output=T> + Sub<Output=T>
     + Mul<Output=T> + Div<Output=T>
     + Rem<Output=T> + Debug
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
    use core::option::Option;
    use core::option::Option::{Some, None};
    use core::num::Float;

    #[test]
    fn from_str_issue7588() {
        let u : Option<u8> = u8::from_str_radix("1000", 10).ok();
        assert_eq!(u, None);
        let s : Option<i16> = i16::from_str_radix("80000", 10).ok();
        assert_eq!(s, None);
    }

    #[test]
    fn test_int_from_str_overflow() {
        let mut i8_val: i8 = 127;
        assert_eq!("127".parse::<i8>().ok(), Some(i8_val));
        assert_eq!("128".parse::<i8>().ok(), None);

        i8_val = i8_val.wrapping_add(1);
        assert_eq!("-128".parse::<i8>().ok(), Some(i8_val));
        assert_eq!("-129".parse::<i8>().ok(), None);

        let mut i16_val: i16 = 32_767;
        assert_eq!("32767".parse::<i16>().ok(), Some(i16_val));
        assert_eq!("32768".parse::<i16>().ok(), None);

        i16_val = i16_val.wrapping_add(1);
        assert_eq!("-32768".parse::<i16>().ok(), Some(i16_val));
        assert_eq!("-32769".parse::<i16>().ok(), None);

        let mut i32_val: i32 = 2_147_483_647;
        assert_eq!("2147483647".parse::<i32>().ok(), Some(i32_val));
        assert_eq!("2147483648".parse::<i32>().ok(), None);

        i32_val = i32_val.wrapping_add(1);
        assert_eq!("-2147483648".parse::<i32>().ok(), Some(i32_val));
        assert_eq!("-2147483649".parse::<i32>().ok(), None);

        let mut i64_val: i64 = 9_223_372_036_854_775_807;
        assert_eq!("9223372036854775807".parse::<i64>().ok(), Some(i64_val));
        assert_eq!("9223372036854775808".parse::<i64>().ok(), None);

        i64_val = i64_val.wrapping_add(1);
        assert_eq!("-9223372036854775808".parse::<i64>().ok(), Some(i64_val));
        assert_eq!("-9223372036854775809".parse::<i64>().ok(), None);
    }

    #[test]
    fn test_leading_plus() {
        assert_eq!("+127".parse::<u8>().ok(), Some(127u8));
        assert_eq!("+9223372036854775807".parse::<i64>().ok(), Some(9223372036854775807i64));
    }

    #[test]
    fn test_invalid() {
        assert_eq!("--129".parse::<i8>().ok(), None);
        assert_eq!("++129".parse::<i8>().ok(), None);
        assert_eq!("Съешь".parse::<u8>().ok(), None);
    }

    #[test]
    fn test_empty() {
        assert_eq!("-".parse::<i8>().ok(), None);
        assert_eq!("+".parse::<i8>().ok(), None);
        assert_eq!("".parse::<u8>().ok(), None);
    }

    macro_rules! test_impl_from {
        ($fn_name: ident, $Small: ty, $Large: ty) => {
            #[test]
            fn $fn_name() {
                let small_max = <$Small>::max_value();
                let small_min = <$Small>::min_value();
                let large_max: $Large = small_max.into();
                let large_min: $Large = small_min.into();
                assert_eq!(large_max as $Small, small_max);
                assert_eq!(large_min as $Small, small_min);
            }
        }
    }

    // Unsigned -> Unsigned
    test_impl_from! { test_u8u16, u8, u16 }
    test_impl_from! { test_u8u32, u8, u32 }
    test_impl_from! { test_u8u64, u8, u64 }
    test_impl_from! { test_u8usize, u8, usize }
    test_impl_from! { test_u16u32, u16, u32 }
    test_impl_from! { test_u16u64, u16, u64 }
    test_impl_from! { test_u32u64, u32, u64 }

    // Signed -> Signed
    test_impl_from! { test_i8i16, i8, i16 }
    test_impl_from! { test_i8i32, i8, i32 }
    test_impl_from! { test_i8i64, i8, i64 }
    test_impl_from! { test_i8isize, i8, isize }
    test_impl_from! { test_i16i32, i16, i32 }
    test_impl_from! { test_i16i64, i16, i64 }
    test_impl_from! { test_i32i64, i32, i64 }

    // Unsigned -> Signed
    test_impl_from! { test_u8i16, u8, i16 }
    test_impl_from! { test_u8i32, u8, i32 }
    test_impl_from! { test_u8i64, u8, i64 }
    test_impl_from! { test_u16i32, u16, i32 }
    test_impl_from! { test_u16i64, u16, i64 }
    test_impl_from! { test_u32i64, u32, i64 }

    // Signed -> Float
    test_impl_from! { test_i8f32, i8, f32 }
    test_impl_from! { test_i8f64, i8, f64 }
    test_impl_from! { test_i16f32, i16, f32 }
    test_impl_from! { test_i16f64, i16, f64 }
    test_impl_from! { test_i32f64, i32, f64 }

    // Unsigned -> Float
    test_impl_from! { test_u8f32, u8, f32 }
    test_impl_from! { test_u8f64, u8, f64 }
    test_impl_from! { test_u16f32, u16, f32 }
    test_impl_from! { test_u16f64, u16, f64 }
    test_impl_from! { test_u32f64, u32, f64 }

    // Float -> Float
    #[test]
    fn test_f32f64() {
        use core::f32;

        let max: f64 = f32::MAX.into();
        assert_eq!(max as f32, f32::MAX);
        assert!(max.is_normal());

        let min: f64 = f32::MIN.into();
        assert_eq!(min as f32, f32::MIN);
        assert!(min.is_normal());

        let min_positive: f64 = f32::MIN_POSITIVE.into();
        assert_eq!(min_positive as f32, f32::MIN_POSITIVE);
        assert!(min_positive.is_normal());

        let epsilon: f64 = f32::EPSILON.into();
        assert_eq!(epsilon as f32, f32::EPSILON);
        assert!(epsilon.is_normal());

        let zero: f64 = (0.0f32).into();
        assert_eq!(zero as f32, 0.0f32);
        assert!(zero.is_sign_positive());

        let neg_zero: f64 = (-0.0f32).into();
        assert_eq!(neg_zero as f32, -0.0f32);
        assert!(neg_zero.is_sign_negative());

        let infinity: f64 = f32::INFINITY.into();
        assert_eq!(infinity as f32, f32::INFINITY);
        assert!(infinity.is_infinite());
        assert!(infinity.is_sign_positive());

        let neg_infinity: f64 = f32::NEG_INFINITY.into();
        assert_eq!(neg_infinity as f32, f32::NEG_INFINITY);
        assert!(neg_infinity.is_infinite());
        assert!(neg_infinity.is_sign_negative());

        let nan: f64 = f32::NAN.into();
        assert!(nan.is_nan());
    }
}
