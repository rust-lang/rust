use core::fmt::Debug;
use core::num::{IntErrorKind, ParseIntError, TryFromIntError, can_not_overflow};
use core::ops::{Add, Div, Mul, Rem, Sub};
use core::str::FromStr;

#[macro_use]
mod int_macros;

mod i128;
mod i16;
mod i32;
mod i64;
mod i8;

#[macro_use]
mod uint_macros;

mod u128;
mod u16;
mod u32;
mod u64;
mod u8;

mod bignum;

mod const_from;
mod dec2flt;
mod flt2dec;
mod int_log;
mod int_sqrt;
mod midpoint;
mod ops;
mod wrapping;

mod float_iter_sum_identity;
mod ieee754;
mod nan;

/// Adds the attribute to all items in the block.
macro_rules! cfg_block {
    ($(#[$attr:meta]{$($it:item)*})*) => {$($(
        #[$attr]
        $it
    )*)*}
}

/// Groups items that assume the pointer width is either 16/32/64, and has to be altered if
/// support for larger/smaller pointer widths are added in the future.
macro_rules! assume_usize_width {
    {$($it:item)*} => {#[cfg(not(any(
        target_pointer_width = "16", target_pointer_width = "32", target_pointer_width = "64")))]
           compile_error!("The current tests of try_from on usize/isize assume that \
                           the pointer width is either 16, 32, or 64");
                    $($it)*
    }
}

/// Helper function for testing numeric operations
pub fn test_num<T>(ten: T, two: T)
where
    T: PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Rem<Output = T>
        + Debug
        + Copy,
{
    assert_eq!(ten.add(two), ten + two);
    assert_eq!(ten.sub(two), ten - two);
    assert_eq!(ten.mul(two), ten * two);
    assert_eq!(ten.div(two), ten / two);
    assert_eq!(ten.rem(two), ten % two);
}

/// Helper function for asserting number parsing returns a specific error
fn test_parse<T>(num_str: &str, expected: Result<T, IntErrorKind>)
where
    T: FromStr<Err = ParseIntError>,
    Result<T, IntErrorKind>: PartialEq + Debug,
{
    assert_eq!(num_str.parse::<T>().map_err(|e| e.kind().clone()), expected)
}

#[test]
fn from_str_issue7588() {
    let u: Option<u8> = u8::from_str_radix("1000", 10).ok();
    assert_eq!(u, None);
    let s: Option<i16> = i16::from_str_radix("80000", 10).ok();
    assert_eq!(s, None);
}

#[test]
fn test_int_from_str_overflow() {
    test_parse::<i8>("127", Ok(127));
    test_parse::<i8>("128", Err(IntErrorKind::PosOverflow));

    test_parse::<i8>("-128", Ok(-128));
    test_parse::<i8>("-129", Err(IntErrorKind::NegOverflow));

    test_parse::<i16>("32767", Ok(32_767));
    test_parse::<i16>("32768", Err(IntErrorKind::PosOverflow));

    test_parse::<i16>("-32768", Ok(-32_768));
    test_parse::<i16>("-32769", Err(IntErrorKind::NegOverflow));

    test_parse::<i32>("2147483647", Ok(2_147_483_647));
    test_parse::<i32>("2147483648", Err(IntErrorKind::PosOverflow));

    test_parse::<i32>("-2147483648", Ok(-2_147_483_648));
    test_parse::<i32>("-2147483649", Err(IntErrorKind::NegOverflow));

    test_parse::<i64>("9223372036854775807", Ok(9_223_372_036_854_775_807));
    test_parse::<i64>("9223372036854775808", Err(IntErrorKind::PosOverflow));

    test_parse::<i64>("-9223372036854775808", Ok(-9_223_372_036_854_775_808));
    test_parse::<i64>("-9223372036854775809", Err(IntErrorKind::NegOverflow));
}

#[test]
fn test_can_not_overflow() {
    fn can_overflow<T>(radix: u32, input: &str) -> bool
    where
        T: std::convert::TryFrom<i8>,
    {
        !can_not_overflow::<T>(radix, T::try_from(-1_i8).is_ok(), input.as_bytes())
    }

    // Positive tests:
    assert!(!can_overflow::<i8>(16, "F"));
    assert!(!can_overflow::<u8>(16, "FF"));

    assert!(!can_overflow::<i8>(10, "9"));
    assert!(!can_overflow::<u8>(10, "99"));

    // Negative tests:

    // Not currently in std lib (issue: #27728)
    fn format_radix<T>(mut x: T, radix: T) -> String
    where
        T: std::ops::Rem<Output = T>,
        T: std::ops::Div<Output = T>,
        T: std::cmp::PartialEq,
        T: std::default::Default,
        T: Copy,
        T: Default,
        u32: TryFrom<T>,
    {
        let mut result = vec![];

        loop {
            let m = x % radix;
            x = x / radix;
            result.push(
                std::char::from_digit(m.try_into().ok().unwrap(), radix.try_into().ok().unwrap())
                    .unwrap(),
            );
            if x == T::default() {
                break;
            }
        }
        result.into_iter().rev().collect()
    }

    macro_rules! check {
        ($($t:ty)*) => ($(
        for base in 2..=36 {
            let num = (<$t>::MAX as u128) + 1;

           // Calculate the string length for the smallest overflowing number:
           let max_len_string = format_radix(num, base as u128);
           // Ensure that string length is deemed to potentially overflow:
           assert!(can_overflow::<$t>(base, &max_len_string));
        }
        )*)
    }

    check! { i8 i16 i32 i64 i128 isize usize u8 u16 u32 u64 }

    // Check u128 separately:
    for base in 2..=36 {
        let num = <u128>::MAX;
        let max_len_string = format_radix(num, base as u128);
        // base 16 fits perfectly for u128 and won't overflow:
        assert_eq!(can_overflow::<u128>(base, &max_len_string), base != 16);
    }
}

#[test]
fn test_leading_plus() {
    test_parse::<u8>("+127", Ok(127));
    test_parse::<i64>("+9223372036854775807", Ok(9223372036854775807));
}

#[test]
fn test_invalid() {
    test_parse::<i8>("--129", Err(IntErrorKind::InvalidDigit));
    test_parse::<i8>("++129", Err(IntErrorKind::InvalidDigit));
    test_parse::<u8>("Съешь", Err(IntErrorKind::InvalidDigit));
    test_parse::<u8>("123Hello", Err(IntErrorKind::InvalidDigit));
    test_parse::<i8>("--", Err(IntErrorKind::InvalidDigit));
    test_parse::<i8>("-", Err(IntErrorKind::InvalidDigit));
    test_parse::<i8>("+", Err(IntErrorKind::InvalidDigit));
    test_parse::<u8>("-1", Err(IntErrorKind::InvalidDigit));
}

#[test]
fn test_empty() {
    test_parse::<u8>("", Err(IntErrorKind::Empty));
}

#[test]
fn test_infallible_try_from_int_error() {
    let func = |x: i8| -> Result<i32, TryFromIntError> { Ok(x.try_into()?) };

    assert!(func(0).is_ok());
}

const _TEST_CONST_PARSE: () = {
    let Ok(-0x8000) = i16::from_str_radix("-8000", 16) else { panic!() };
    let Ok(12345) = u64::from_str_radix("12345", 10) else { panic!() };
    if let Err(e) = i8::from_str_radix("+", 10) {
        let IntErrorKind::InvalidDigit = e.kind() else { panic!() };
    } else {
        panic!()
    }
};

macro_rules! test_impl_from {
    ($fn_name:ident, bool, $target: ty) => {
        #[test]
        fn $fn_name() {
            let one: $target = 1;
            let zero: $target = 0;
            assert_eq!(one, <$target>::from(true));
            assert_eq!(zero, <$target>::from(false));
        }
    };
    ($fn_name: ident, $Small: ty, $Large: ty) => {
        #[test]
        fn $fn_name() {
            let small_max = <$Small>::MAX;
            let small_min = <$Small>::MIN;
            let large_max: $Large = small_max.into();
            let large_min: $Large = small_min.into();
            assert_eq!(large_max as $Small, small_max);
            assert_eq!(large_min as $Small, small_min);
        }
    };
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

// Bool -> Integer
test_impl_from! { test_boolu8, bool, u8 }
test_impl_from! { test_boolu16, bool, u16 }
test_impl_from! { test_boolu32, bool, u32 }
test_impl_from! { test_boolu64, bool, u64 }
test_impl_from! { test_boolu128, bool, u128 }
test_impl_from! { test_booli8, bool, i8 }
test_impl_from! { test_booli16, bool, i16 }
test_impl_from! { test_booli32, bool, i32 }
test_impl_from! { test_booli64, bool, i64 }
test_impl_from! { test_booli128, bool, i128 }

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

/// Conversions where the full width of $source can be represented as $target
macro_rules! test_impl_try_from_always_ok {
    ($fn_name:ident, $source:ty, $target: ty) => {
        #[test]
        fn $fn_name() {
            let max = <$source>::MAX;
            let min = <$source>::MIN;
            let zero: $source = 0;
            assert_eq!(<$target as TryFrom<$source>>::try_from(max).unwrap(), max as $target);
            assert_eq!(<$target as TryFrom<$source>>::try_from(min).unwrap(), min as $target);
            assert_eq!(<$target as TryFrom<$source>>::try_from(zero).unwrap(), zero as $target);
        }
    };
}

test_impl_try_from_always_ok! { test_try_u8u8, u8, u8 }
test_impl_try_from_always_ok! { test_try_u8u16, u8, u16 }
test_impl_try_from_always_ok! { test_try_u8u32, u8, u32 }
test_impl_try_from_always_ok! { test_try_u8u64, u8, u64 }
test_impl_try_from_always_ok! { test_try_u8u128, u8, u128 }
test_impl_try_from_always_ok! { test_try_u8i16, u8, i16 }
test_impl_try_from_always_ok! { test_try_u8i32, u8, i32 }
test_impl_try_from_always_ok! { test_try_u8i64, u8, i64 }
test_impl_try_from_always_ok! { test_try_u8i128, u8, i128 }

test_impl_try_from_always_ok! { test_try_u16u16, u16, u16 }
test_impl_try_from_always_ok! { test_try_u16u32, u16, u32 }
test_impl_try_from_always_ok! { test_try_u16u64, u16, u64 }
test_impl_try_from_always_ok! { test_try_u16u128, u16, u128 }
test_impl_try_from_always_ok! { test_try_u16i32, u16, i32 }
test_impl_try_from_always_ok! { test_try_u16i64, u16, i64 }
test_impl_try_from_always_ok! { test_try_u16i128, u16, i128 }

test_impl_try_from_always_ok! { test_try_u32u32, u32, u32 }
test_impl_try_from_always_ok! { test_try_u32u64, u32, u64 }
test_impl_try_from_always_ok! { test_try_u32u128, u32, u128 }
test_impl_try_from_always_ok! { test_try_u32i64, u32, i64 }
test_impl_try_from_always_ok! { test_try_u32i128, u32, i128 }

test_impl_try_from_always_ok! { test_try_u64u64, u64, u64 }
test_impl_try_from_always_ok! { test_try_u64u128, u64, u128 }
test_impl_try_from_always_ok! { test_try_u64i128, u64, i128 }

test_impl_try_from_always_ok! { test_try_u128u128, u128, u128 }

test_impl_try_from_always_ok! { test_try_i8i8, i8, i8 }
test_impl_try_from_always_ok! { test_try_i8i16, i8, i16 }
test_impl_try_from_always_ok! { test_try_i8i32, i8, i32 }
test_impl_try_from_always_ok! { test_try_i8i64, i8, i64 }
test_impl_try_from_always_ok! { test_try_i8i128, i8, i128 }

test_impl_try_from_always_ok! { test_try_i16i16, i16, i16 }
test_impl_try_from_always_ok! { test_try_i16i32, i16, i32 }
test_impl_try_from_always_ok! { test_try_i16i64, i16, i64 }
test_impl_try_from_always_ok! { test_try_i16i128, i16, i128 }

test_impl_try_from_always_ok! { test_try_i32i32, i32, i32 }
test_impl_try_from_always_ok! { test_try_i32i64, i32, i64 }
test_impl_try_from_always_ok! { test_try_i32i128, i32, i128 }

test_impl_try_from_always_ok! { test_try_i64i64, i64, i64 }
test_impl_try_from_always_ok! { test_try_i64i128, i64, i128 }

test_impl_try_from_always_ok! { test_try_i128i128, i128, i128 }

test_impl_try_from_always_ok! { test_try_usizeusize, usize, usize }
test_impl_try_from_always_ok! { test_try_isizeisize, isize, isize }

assume_usize_width! {
    test_impl_try_from_always_ok! { test_try_u8usize, u8, usize }
    test_impl_try_from_always_ok! { test_try_u8isize, u8, isize }
    test_impl_try_from_always_ok! { test_try_i8isize, i8, isize }

    test_impl_try_from_always_ok! { test_try_u16usize, u16, usize }
    test_impl_try_from_always_ok! { test_try_i16isize, i16, isize }

    test_impl_try_from_always_ok! { test_try_usizeu64, usize, u64 }
    test_impl_try_from_always_ok! { test_try_usizeu128, usize, u128 }
    test_impl_try_from_always_ok! { test_try_usizei128, usize, i128 }

    test_impl_try_from_always_ok! { test_try_isizei64, isize, i64 }
    test_impl_try_from_always_ok! { test_try_isizei128, isize, i128 }

    cfg_block!(
        #[cfg(target_pointer_width = "16")] {
            test_impl_try_from_always_ok! { test_try_usizeu16, usize, u16 }
            test_impl_try_from_always_ok! { test_try_isizei16, isize, i16 }
            test_impl_try_from_always_ok! { test_try_usizeu32, usize, u32 }
            test_impl_try_from_always_ok! { test_try_usizei32, usize, i32 }
            test_impl_try_from_always_ok! { test_try_isizei32, isize, i32 }
            test_impl_try_from_always_ok! { test_try_usizei64, usize, i64 }
        }

        #[cfg(target_pointer_width = "32")] {
            test_impl_try_from_always_ok! { test_try_u16isize, u16, isize }
            test_impl_try_from_always_ok! { test_try_usizeu32, usize, u32 }
            test_impl_try_from_always_ok! { test_try_isizei32, isize, i32 }
            test_impl_try_from_always_ok! { test_try_u32usize, u32, usize }
            test_impl_try_from_always_ok! { test_try_i32isize, i32, isize }
            test_impl_try_from_always_ok! { test_try_usizei64, usize, i64 }
        }

        #[cfg(target_pointer_width = "64")] {
            test_impl_try_from_always_ok! { test_try_u16isize, u16, isize }
            test_impl_try_from_always_ok! { test_try_u32usize, u32, usize }
            test_impl_try_from_always_ok! { test_try_u32isize, u32, isize }
            test_impl_try_from_always_ok! { test_try_i32isize, i32, isize }
            test_impl_try_from_always_ok! { test_try_u64usize, u64, usize }
            test_impl_try_from_always_ok! { test_try_i64isize, i64, isize }
        }
    );
}

/// Conversions where max of $source can be represented as $target,
macro_rules! test_impl_try_from_signed_to_unsigned_upper_ok {
    ($fn_name:ident, $source:ty, $target:ty) => {
        #[test]
        fn $fn_name() {
            let max = <$source>::MAX;
            let min = <$source>::MIN;
            let zero: $source = 0;
            let neg_one: $source = -1;
            assert_eq!(<$target as TryFrom<$source>>::try_from(max).unwrap(), max as $target);
            assert!(<$target as TryFrom<$source>>::try_from(min).is_err());
            assert_eq!(<$target as TryFrom<$source>>::try_from(zero).unwrap(), zero as $target);
            assert!(<$target as TryFrom<$source>>::try_from(neg_one).is_err());
        }
    };
}

test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i8u8, i8, u8 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i8u16, i8, u16 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i8u32, i8, u32 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i8u64, i8, u64 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i8u128, i8, u128 }

test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i16u16, i16, u16 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i16u32, i16, u32 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i16u64, i16, u64 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i16u128, i16, u128 }

test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i32u32, i32, u32 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i32u64, i32, u64 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i32u128, i32, u128 }

test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i64u64, i64, u64 }
test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i64u128, i64, u128 }

test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i128u128, i128, u128 }

assume_usize_width! {
    test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i8usize, i8, usize }
    test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i16usize, i16, usize }

    test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_isizeu64, isize, u64 }
    test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_isizeu128, isize, u128 }
    test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_isizeusize, isize, usize }

    cfg_block!(
        #[cfg(target_pointer_width = "16")] {
            test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_isizeu16, isize, u16 }
            test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_isizeu32, isize, u32 }
        }

        #[cfg(target_pointer_width = "32")] {
            test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_isizeu32, isize, u32 }

            test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i32usize, i32, usize }
        }

        #[cfg(target_pointer_width = "64")] {
            test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i32usize, i32, usize }
            test_impl_try_from_signed_to_unsigned_upper_ok! { test_try_i64usize, i64, usize }
        }
    );
}

/// Conversions where max of $source can not be represented as $target,
/// but min can.
macro_rules! test_impl_try_from_unsigned_to_signed_upper_err {
    ($fn_name:ident, $source:ty, $target:ty) => {
        #[test]
        fn $fn_name() {
            let max = <$source>::MAX;
            let min = <$source>::MIN;
            let zero: $source = 0;
            assert!(<$target as TryFrom<$source>>::try_from(max).is_err());
            assert_eq!(<$target as TryFrom<$source>>::try_from(min).unwrap(), min as $target);
            assert_eq!(<$target as TryFrom<$source>>::try_from(zero).unwrap(), zero as $target);
        }
    };
}

test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u8i8, u8, i8 }

test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u16i8, u16, i8 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u16i16, u16, i16 }

test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u32i8, u32, i8 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u32i16, u32, i16 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u32i32, u32, i32 }

test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u64i8, u64, i8 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u64i16, u64, i16 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u64i32, u64, i32 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u64i64, u64, i64 }

test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u128i8, u128, i8 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u128i16, u128, i16 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u128i32, u128, i32 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u128i64, u128, i64 }
test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u128i128, u128, i128 }

assume_usize_width! {
    test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u64isize, u64, isize }
    test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u128isize, u128, isize }

    test_impl_try_from_unsigned_to_signed_upper_err! { test_try_usizei8, usize, i8 }
    test_impl_try_from_unsigned_to_signed_upper_err! { test_try_usizei16, usize, i16 }
    test_impl_try_from_unsigned_to_signed_upper_err! { test_try_usizeisize, usize, isize }

    cfg_block!(
        #[cfg(target_pointer_width = "16")] {
            test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u16isize, u16, isize }
            test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u32isize, u32, isize }
        }

        #[cfg(target_pointer_width = "32")] {
            test_impl_try_from_unsigned_to_signed_upper_err! { test_try_u32isize, u32, isize }
            test_impl_try_from_unsigned_to_signed_upper_err! { test_try_usizei32, usize, i32 }
        }

        #[cfg(target_pointer_width = "64")] {
            test_impl_try_from_unsigned_to_signed_upper_err! { test_try_usizei32, usize, i32 }
            test_impl_try_from_unsigned_to_signed_upper_err! { test_try_usizei64, usize, i64 }
        }
    );
}

/// Conversions where min/max of $source can not be represented as $target.
macro_rules! test_impl_try_from_same_sign_err {
    ($fn_name:ident, $source:ty, $target:ty) => {
        #[test]
        fn $fn_name() {
            let max = <$source>::MAX;
            let min = <$source>::MIN;
            let zero: $source = 0;
            let t_max = <$target>::MAX;
            let t_min = <$target>::MIN;
            assert!(<$target as TryFrom<$source>>::try_from(max).is_err());
            if min != 0 {
                assert!(<$target as TryFrom<$source>>::try_from(min).is_err());
            }
            assert_eq!(<$target as TryFrom<$source>>::try_from(zero).unwrap(), zero as $target);
            assert_eq!(
                <$target as TryFrom<$source>>::try_from(t_max as $source).unwrap(),
                t_max as $target
            );
            assert_eq!(
                <$target as TryFrom<$source>>::try_from(t_min as $source).unwrap(),
                t_min as $target
            );
        }
    };
}

test_impl_try_from_same_sign_err! { test_try_u16u8, u16, u8 }

test_impl_try_from_same_sign_err! { test_try_u32u8, u32, u8 }
test_impl_try_from_same_sign_err! { test_try_u32u16, u32, u16 }

test_impl_try_from_same_sign_err! { test_try_u64u8, u64, u8 }
test_impl_try_from_same_sign_err! { test_try_u64u16, u64, u16 }
test_impl_try_from_same_sign_err! { test_try_u64u32, u64, u32 }

test_impl_try_from_same_sign_err! { test_try_u128u8, u128, u8 }
test_impl_try_from_same_sign_err! { test_try_u128u16, u128, u16 }
test_impl_try_from_same_sign_err! { test_try_u128u32, u128, u32 }
test_impl_try_from_same_sign_err! { test_try_u128u64, u128, u64 }

test_impl_try_from_same_sign_err! { test_try_i16i8, i16, i8 }
test_impl_try_from_same_sign_err! { test_try_isizei8, isize, i8 }

test_impl_try_from_same_sign_err! { test_try_i32i8, i32, i8 }
test_impl_try_from_same_sign_err! { test_try_i32i16, i32, i16 }

test_impl_try_from_same_sign_err! { test_try_i64i8, i64, i8 }
test_impl_try_from_same_sign_err! { test_try_i64i16, i64, i16 }
test_impl_try_from_same_sign_err! { test_try_i64i32, i64, i32 }

test_impl_try_from_same_sign_err! { test_try_i128i8, i128, i8 }
test_impl_try_from_same_sign_err! { test_try_i128i16, i128, i16 }
test_impl_try_from_same_sign_err! { test_try_i128i32, i128, i32 }
test_impl_try_from_same_sign_err! { test_try_i128i64, i128, i64 }

assume_usize_width! {
    test_impl_try_from_same_sign_err! { test_try_usizeu8, usize, u8 }
    test_impl_try_from_same_sign_err! { test_try_u128usize, u128, usize }
    test_impl_try_from_same_sign_err! { test_try_i128isize, i128, isize }

    cfg_block!(
        #[cfg(target_pointer_width = "16")] {
            test_impl_try_from_same_sign_err! { test_try_u32usize, u32, usize }
            test_impl_try_from_same_sign_err! { test_try_u64usize, u64, usize }

            test_impl_try_from_same_sign_err! { test_try_i32isize, i32, isize }
            test_impl_try_from_same_sign_err! { test_try_i64isize, i64, isize }
        }

        #[cfg(target_pointer_width = "32")] {
            test_impl_try_from_same_sign_err! { test_try_u64usize, u64, usize }
            test_impl_try_from_same_sign_err! { test_try_usizeu16, usize, u16 }

            test_impl_try_from_same_sign_err! { test_try_i64isize, i64, isize }
            test_impl_try_from_same_sign_err! { test_try_isizei16, isize, i16 }
        }

        #[cfg(target_pointer_width = "64")] {
            test_impl_try_from_same_sign_err! { test_try_usizeu16, usize, u16 }
            test_impl_try_from_same_sign_err! { test_try_usizeu32, usize, u32 }

            test_impl_try_from_same_sign_err! { test_try_isizei16, isize, i16 }
            test_impl_try_from_same_sign_err! { test_try_isizei32, isize, i32 }
        }
    );
}

/// Conversions where neither the min nor the max of $source can be represented by
/// $target, but max/min of the target can be represented by the source.
macro_rules! test_impl_try_from_signed_to_unsigned_err {
    ($fn_name:ident, $source:ty, $target:ty) => {
        #[test]
        fn $fn_name() {
            let max = <$source>::MAX;
            let min = <$source>::MIN;
            let zero: $source = 0;
            let t_max = <$target>::MAX;
            let t_min = <$target>::MIN;
            assert!(<$target as TryFrom<$source>>::try_from(max).is_err());
            assert!(<$target as TryFrom<$source>>::try_from(min).is_err());
            assert_eq!(<$target as TryFrom<$source>>::try_from(zero).unwrap(), zero as $target);
            assert_eq!(
                <$target as TryFrom<$source>>::try_from(t_max as $source).unwrap(),
                t_max as $target
            );
            assert_eq!(
                <$target as TryFrom<$source>>::try_from(t_min as $source).unwrap(),
                t_min as $target
            );
        }
    };
}

test_impl_try_from_signed_to_unsigned_err! { test_try_i16u8, i16, u8 }

test_impl_try_from_signed_to_unsigned_err! { test_try_i32u8, i32, u8 }
test_impl_try_from_signed_to_unsigned_err! { test_try_i32u16, i32, u16 }

test_impl_try_from_signed_to_unsigned_err! { test_try_i64u8, i64, u8 }
test_impl_try_from_signed_to_unsigned_err! { test_try_i64u16, i64, u16 }
test_impl_try_from_signed_to_unsigned_err! { test_try_i64u32, i64, u32 }

test_impl_try_from_signed_to_unsigned_err! { test_try_i128u8, i128, u8 }
test_impl_try_from_signed_to_unsigned_err! { test_try_i128u16, i128, u16 }
test_impl_try_from_signed_to_unsigned_err! { test_try_i128u32, i128, u32 }
test_impl_try_from_signed_to_unsigned_err! { test_try_i128u64, i128, u64 }

assume_usize_width! {
    test_impl_try_from_signed_to_unsigned_err! { test_try_isizeu8, isize, u8 }
    test_impl_try_from_signed_to_unsigned_err! { test_try_i128usize, i128, usize }

    cfg_block! {
        #[cfg(target_pointer_width = "16")] {
            test_impl_try_from_signed_to_unsigned_err! { test_try_i32usize, i32, usize }
            test_impl_try_from_signed_to_unsigned_err! { test_try_i64usize, i64, usize }
        }
        #[cfg(target_pointer_width = "32")] {
            test_impl_try_from_signed_to_unsigned_err! { test_try_i64usize, i64, usize }

            test_impl_try_from_signed_to_unsigned_err! { test_try_isizeu16, isize, u16 }
        }
        #[cfg(target_pointer_width = "64")] {
            test_impl_try_from_signed_to_unsigned_err! { test_try_isizeu16, isize, u16 }
            test_impl_try_from_signed_to_unsigned_err! { test_try_isizeu32, isize, u32 }
        }
    }
}

macro_rules! test_float {
    ($modname: ident, $fassert: ident, $fty: ty) => {
        mod $modname {
            #[test]
            fn min() {
                $fassert!((0.0 as $fty).min(0.0), 0.0);
                $fassert!((0.0 as $fty).min(0.0).is_sign_positive());
                $fassert!((-0.0 as $fty).min(-0.0), -0.0);
                $fassert!((-0.0 as $fty).min(-0.0).is_sign_negative());
                $fassert!((9.0 as $fty).min(9.0), 9.0);
                $fassert!((-9.0 as $fty).min(0.0), -9.0);
                $fassert!((0.0 as $fty).min(9.0), 0.0);
                $fassert!((0.0 as $fty).min(9.0).is_sign_positive());
                $fassert!((-0.0 as $fty).min(9.0), -0.0);
                $fassert!((-0.0 as $fty).min(9.0).is_sign_negative());
                $fassert!((-0.0 as $fty).min(-9.0), -9.0);
                $fassert!(<$fty>::INFINITY.min(9.0), 9.0);
                $fassert!((9.0 as $fty).min(<$fty>::INFINITY), 9.0);
                $fassert!(<$fty>::INFINITY.min(-9.0), -9.0);
                $fassert!((-9.0 as $fty).min(<$fty>::INFINITY), -9.0);
                $fassert!(<$fty>::NEG_INFINITY.min(9.0), <$fty>::NEG_INFINITY);
                $fassert!((9.0 as $fty).min(<$fty>::NEG_INFINITY), <$fty>::NEG_INFINITY);
                $fassert!(<$fty>::NEG_INFINITY.min(-9.0), <$fty>::NEG_INFINITY);
                $fassert!((-9.0 as $fty).min(<$fty>::NEG_INFINITY), <$fty>::NEG_INFINITY);
                $fassert!(<$fty>::NAN.min(9.0), 9.0);
                $fassert!(<$fty>::NAN.min(-9.0), -9.0);
                $fassert!((9.0 as $fty).min(<$fty>::NAN), 9.0);
                $fassert!((-9.0 as $fty).min(<$fty>::NAN), -9.0);
                $fassert!(<$fty>::NAN.min(<$fty>::NAN).is_nan());
            }
            #[test]
            fn max() {
                $fassert!((0.0 as $fty).max(0.0), 0.0);
                $fassert!((0.0 as $fty).max(0.0).is_sign_positive());
                $fassert!((-0.0 as $fty).max(-0.0), -0.0);
                $fassert!((-0.0 as $fty).max(-0.0).is_sign_negative());
                $fassert!((9.0 as $fty).max(9.0), 9.0);
                $fassert!((-9.0 as $fty).max(0.0), 0.0);
                $fassert!((-9.0 as $fty).max(0.0).is_sign_positive());
                $fassert!((-9.0 as $fty).max(-0.0), -0.0);
                $fassert!((-9.0 as $fty).max(-0.0).is_sign_negative());
                $fassert!((0.0 as $fty).max(9.0), 9.0);
                $fassert!((0.0 as $fty).max(-9.0), 0.0);
                $fassert!((0.0 as $fty).max(-9.0).is_sign_positive());
                $fassert!((-0.0 as $fty).max(-9.0), -0.0);
                $fassert!((-0.0 as $fty).max(-9.0).is_sign_negative());
                $fassert!(<$fty>::INFINITY.max(9.0), <$fty>::INFINITY);
                $fassert!((9.0 as $fty).max(<$fty>::INFINITY), <$fty>::INFINITY);
                $fassert!(<$fty>::INFINITY.max(-9.0), <$fty>::INFINITY);
                $fassert!((-9.0 as $fty).max(<$fty>::INFINITY), <$fty>::INFINITY);
                $fassert!(<$fty>::NEG_INFINITY.max(9.0), 9.0);
                $fassert!((9.0 as $fty).max(<$fty>::NEG_INFINITY), 9.0);
                $fassert!(<$fty>::NEG_INFINITY.max(-9.0), -9.0);
                $fassert!((-9.0 as $fty).max(<$fty>::NEG_INFINITY), -9.0);
                $fassert!(<$fty>::NAN.max(9.0), 9.0);
                $fassert!(<$fty>::NAN.max(-9.0), -9.0);
                $fassert!((9.0 as $fty).max(<$fty>::NAN), 9.0);
                $fassert!((-9.0 as $fty).max(<$fty>::NAN), -9.0);
                $fassert!(<$fty>::NAN.max(<$fty>::NAN).is_nan());
            }
            #[test]
            fn minimum() {
                $fassert!((0.0 as $fty).minimum(0.0), 0.0);
                $fassert!((0.0 as $fty).minimum(0.0).is_sign_positive());
                $fassert!((-0.0 as $fty).minimum(0.0), -0.0);
                $fassert!((-0.0 as $fty).minimum(0.0).is_sign_negative());
                $fassert!((-0.0 as $fty).minimum(-0.0), -0.0);
                $fassert!((-0.0 as $fty).minimum(-0.0).is_sign_negative());
                $fassert!((9.0 as $fty).minimum(9.0), 9.0);
                $fassert!((-9.0 as $fty).minimum(0.0), -9.0);
                $fassert!((0.0 as $fty).minimum(9.0), 0.0);
                $fassert!((0.0 as $fty).minimum(9.0).is_sign_positive());
                $fassert!((-0.0 as $fty).minimum(9.0), -0.0);
                $fassert!((-0.0 as $fty).minimum(9.0).is_sign_negative());
                $fassert!((-0.0 as $fty).minimum(-9.0), -9.0);
                $fassert!(<$fty>::INFINITY.minimum(9.0), 9.0);
                $fassert!((9.0 as $fty).minimum(<$fty>::INFINITY), 9.0);
                $fassert!(<$fty>::INFINITY.minimum(-9.0), -9.0);
                $fassert!((-9.0 as $fty).minimum(<$fty>::INFINITY), -9.0);
                $fassert!(<$fty>::NEG_INFINITY.minimum(9.0), <$fty>::NEG_INFINITY);
                $fassert!((9.0 as $fty).minimum(<$fty>::NEG_INFINITY), <$fty>::NEG_INFINITY);
                $fassert!(<$fty>::NEG_INFINITY.minimum(-9.0), <$fty>::NEG_INFINITY);
                $fassert!((-9.0 as $fty).minimum(<$fty>::NEG_INFINITY), <$fty>::NEG_INFINITY);
                $fassert!(<$fty>::NAN.minimum(9.0).is_nan());
                $fassert!(<$fty>::NAN.minimum(-9.0).is_nan());
                $fassert!((9.0 as $fty).minimum(<$fty>::NAN).is_nan());
                $fassert!((-9.0 as $fty).minimum(<$fty>::NAN).is_nan());
                $fassert!(<$fty>::NAN.minimum(<$fty>::NAN).is_nan());
            }
            #[test]
            fn maximum() {
                $fassert!((0.0 as $fty).maximum(0.0), 0.0);
                $fassert!((0.0 as $fty).maximum(0.0).is_sign_positive());
                $fassert!((-0.0 as $fty).maximum(0.0), 0.0);
                $fassert!((-0.0 as $fty).maximum(0.0).is_sign_positive());
                $fassert!((-0.0 as $fty).maximum(-0.0), -0.0);
                $fassert!((-0.0 as $fty).maximum(-0.0).is_sign_negative());
                $fassert!((9.0 as $fty).maximum(9.0), 9.0);
                $fassert!((-9.0 as $fty).maximum(0.0), 0.0);
                $fassert!((-9.0 as $fty).maximum(0.0).is_sign_positive());
                $fassert!((-9.0 as $fty).maximum(-0.0), -0.0);
                $fassert!((-9.0 as $fty).maximum(-0.0).is_sign_negative());
                $fassert!((0.0 as $fty).maximum(9.0), 9.0);
                $fassert!((0.0 as $fty).maximum(-9.0), 0.0);
                $fassert!((0.0 as $fty).maximum(-9.0).is_sign_positive());
                $fassert!((-0.0 as $fty).maximum(-9.0), -0.0);
                $fassert!((-0.0 as $fty).maximum(-9.0).is_sign_negative());
                $fassert!(<$fty>::INFINITY.maximum(9.0), <$fty>::INFINITY);
                $fassert!((9.0 as $fty).maximum(<$fty>::INFINITY), <$fty>::INFINITY);
                $fassert!(<$fty>::INFINITY.maximum(-9.0), <$fty>::INFINITY);
                $fassert!((-9.0 as $fty).maximum(<$fty>::INFINITY), <$fty>::INFINITY);
                $fassert!(<$fty>::NEG_INFINITY.maximum(9.0), 9.0);
                $fassert!((9.0 as $fty).maximum(<$fty>::NEG_INFINITY), 9.0);
                $fassert!(<$fty>::NEG_INFINITY.maximum(-9.0), -9.0);
                $fassert!((-9.0 as $fty).maximum(<$fty>::NEG_INFINITY), -9.0);
                $fassert!(<$fty>::NAN.maximum(9.0).is_nan());
                $fassert!(<$fty>::NAN.maximum(-9.0).is_nan());
                $fassert!((9.0 as $fty).maximum(<$fty>::NAN).is_nan());
                $fassert!((-9.0 as $fty).maximum(<$fty>::NAN).is_nan());
                $fassert!(<$fty>::NAN.maximum(<$fty>::NAN).is_nan());
            }
            #[test]
            fn midpoint() {
                $fassert!((0.5 as $fty).midpoint(0.5), 0.5);
                $fassert!((0.5 as $fty).midpoint(2.5), 1.5);
                $fassert!((3.0 as $fty).midpoint(4.0), 3.5);
                $fassert!((-3.0 as $fty).midpoint(4.0), 0.5);
                $fassert!((3.0 as $fty).midpoint(-4.0), -0.5);
                $fassert!((-3.0 as $fty).midpoint(-4.0), -3.5);
                $fassert!((0.0 as $fty).midpoint(0.0), 0.0);
                $fassert!((-0.0 as $fty).midpoint(-0.0), -0.0);
                $fassert!((-5.0 as $fty).midpoint(5.0), 0.0);
                $fassert!(<$fty>::MAX.midpoint(<$fty>::MIN), 0.0);
                $fassert!(<$fty>::MIN.midpoint(<$fty>::MAX), -0.0);
                $fassert!(<$fty>::MAX.midpoint(<$fty>::MIN_POSITIVE), <$fty>::MAX / 2.);
                $fassert!((-<$fty>::MAX).midpoint(<$fty>::MIN_POSITIVE), -<$fty>::MAX / 2.);
                $fassert!(<$fty>::MAX.midpoint(-<$fty>::MIN_POSITIVE), <$fty>::MAX / 2.);
                $fassert!((-<$fty>::MAX).midpoint(-<$fty>::MIN_POSITIVE), -<$fty>::MAX / 2.);
                $fassert!((<$fty>::MIN_POSITIVE).midpoint(<$fty>::MAX), <$fty>::MAX / 2.);
                $fassert!((<$fty>::MIN_POSITIVE).midpoint(-<$fty>::MAX), -<$fty>::MAX / 2.);
                $fassert!((-<$fty>::MIN_POSITIVE).midpoint(<$fty>::MAX), <$fty>::MAX / 2.);
                $fassert!((-<$fty>::MIN_POSITIVE).midpoint(-<$fty>::MAX), -<$fty>::MAX / 2.);
                $fassert!(<$fty>::MAX.midpoint(<$fty>::MAX), <$fty>::MAX);
                $fassert!(
                    (<$fty>::MIN_POSITIVE).midpoint(<$fty>::MIN_POSITIVE),
                    <$fty>::MIN_POSITIVE
                );
                $fassert!(
                    (-<$fty>::MIN_POSITIVE).midpoint(-<$fty>::MIN_POSITIVE),
                    -<$fty>::MIN_POSITIVE
                );
                $fassert!(<$fty>::MAX.midpoint(5.0), <$fty>::MAX / 2.0 + 2.5);
                $fassert!(<$fty>::MAX.midpoint(-5.0), <$fty>::MAX / 2.0 - 2.5);
                $fassert!(<$fty>::INFINITY.midpoint(<$fty>::INFINITY), <$fty>::INFINITY);
                $fassert!(
                    <$fty>::NEG_INFINITY.midpoint(<$fty>::NEG_INFINITY),
                    <$fty>::NEG_INFINITY
                );
                $fassert!(<$fty>::NAN.midpoint(1.0).is_nan());
                $fassert!((1.0 as $fty).midpoint(<$fty>::NAN).is_nan());
                $fassert!(<$fty>::NAN.midpoint(<$fty>::NAN).is_nan());

                // test if large differences in magnitude are still correctly computed.
                // NOTE: that because of how small x and y are, x + y can never overflow
                // so (x + y) / 2.0 is always correct
                // in particular, `2.pow(i)` will  never be at the max exponent, so it could
                // be safely doubled, while j is significantly smaller.
                for i in <$fty>::MAX_EXP.saturating_sub(64)..<$fty>::MAX_EXP {
                    for j in 0..64u8 {
                        let large = (2.0 as $fty).powi(i);
                        // a much smaller number, such that there is no chance of overflow to test
                        // potential double rounding in midpoint's implementation.
                        let small = (2.0 as $fty).powi(<$fty>::MAX_EXP - 1)
                            * <$fty>::EPSILON
                            * <$fty>::from(j);

                        let naive = (large + small) / 2.0;
                        let midpoint = large.midpoint(small);

                        assert_eq!(naive, midpoint);
                    }
                }
            }
            #[test]
            fn abs() {
                $fassert!((-1.0 as $fty).abs(), 1.0);
                $fassert!((1.0 as $fty).abs(), 1.0);
                $fassert!(<$fty>::NEG_INFINITY.abs(), <$fty>::INFINITY);
                $fassert!(<$fty>::INFINITY.abs(), <$fty>::INFINITY);
            }
            #[test]
            fn copysign() {
                $fassert!((1.0 as $fty).copysign(-2.0), -1.0);
                $fassert!((-1.0 as $fty).copysign(2.0), 1.0);
                $fassert!(<$fty>::INFINITY.copysign(-0.0), <$fty>::NEG_INFINITY);
                $fassert!(<$fty>::NEG_INFINITY.copysign(0.0), <$fty>::INFINITY);
            }
            #[test]
            fn rem_euclid() {
                // FIXME: Use $fassert when rem_euclid becomes const
                assert!(<$fty>::INFINITY.rem_euclid((42.0 as $fty)).is_nan());
                assert_eq!((42.0 as $fty).rem_euclid(<$fty>::INFINITY), (42.0 as $fty));
                assert!((42.0 as $fty).rem_euclid(<$fty>::NAN).is_nan());
                assert!(<$fty>::INFINITY.rem_euclid(<$fty>::INFINITY).is_nan());
                assert!(<$fty>::INFINITY.rem_euclid(<$fty>::NAN).is_nan());
                assert!(<$fty>::NAN.rem_euclid(<$fty>::INFINITY).is_nan());
            }
            #[test]
            fn div_euclid() {
                // FIXME: Use $fassert when div_euclid becomes const
                assert_eq!((42.0 as $fty).div_euclid(<$fty>::INFINITY), 0.0);
                assert!((42.0 as $fty).div_euclid(<$fty>::NAN).is_nan());
                assert!(<$fty>::INFINITY.div_euclid(<$fty>::INFINITY).is_nan());
                assert!(<$fty>::INFINITY.div_euclid(<$fty>::NAN).is_nan());
                assert!(<$fty>::NAN.div_euclid(<$fty>::INFINITY).is_nan());
            }
        }
    };
}

// Custom assert macro that distribute between assert! and assert_eq! in a non-const context
macro_rules! float_assert {
    ($b:expr) => {
        assert!($b);
    };
    ($left:expr, $right:expr) => {
        assert_eq!($left, $right);
    };
}

// Custom assert macro that only uses assert! in a const context
macro_rules! float_const_assert {
    ($b:expr) => {
        assert!(const { $b });
    };
    ($left:expr, $right:expr) => {
        assert!(const { $left == $right });
    };
}

test_float!(f32, float_assert, f32);
test_float!(f32_const, float_const_assert, f32);
test_float!(f64, float_assert, f64);
test_float!(f64_const, float_const_assert, f64);
