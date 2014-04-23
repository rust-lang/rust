// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]
#![doc(hidden)]

macro_rules! int_module (($T:ty, $bits:expr) => (

// FIXME(#11621): Should be deprecated once CTFE is implemented in favour of
// calling the `mem::size_of` function.
pub static BITS : uint = $bits;
// FIXME(#11621): Should be deprecated once CTFE is implemented in favour of
// calling the `mem::size_of` function.
pub static BYTES : uint = ($bits / 8);

// FIXME(#11621): Should be deprecated once CTFE is implemented in favour of
// calling the `Bounded::min_value` function.
pub static MIN: $T = (-1 as $T) << (BITS - 1);
// FIXME(#9837): Compute MIN like this so the high bits that shouldn't exist are 0.
// FIXME(#11621): Should be deprecated once CTFE is implemented in favour of
// calling the `Bounded::max_value` function.
pub static MAX: $T = !MIN;

impl CheckedDiv for $T {
    #[inline]
    fn checked_div(&self, v: &$T) -> Option<$T> {
        if *v == 0 || (*self == MIN && *v == -1) {
            None
        } else {
            Some(self / *v)
        }
    }
}

impl Num for $T {}

#[cfg(not(test))]
impl Ord for $T {
    #[inline]
    fn lt(&self, other: &$T) -> bool { return (*self) < (*other); }
}

#[cfg(not(test))]
impl Eq for $T {
    #[inline]
    fn eq(&self, other: &$T) -> bool { return (*self) == (*other); }
}

impl Default for $T {
    #[inline]
    fn default() -> $T { 0 }
}

impl Zero for $T {
    #[inline]
    fn zero() -> $T { 0 }

    #[inline]
    fn is_zero(&self) -> bool { *self == 0 }
}

impl One for $T {
    #[inline]
    fn one() -> $T { 1 }
}

#[cfg(not(test))]
impl Add<$T,$T> for $T {
    #[inline]
    fn add(&self, other: &$T) -> $T { *self + *other }
}

#[cfg(not(test))]
impl Sub<$T,$T> for $T {
    #[inline]
    fn sub(&self, other: &$T) -> $T { *self - *other }
}

#[cfg(not(test))]
impl Mul<$T,$T> for $T {
    #[inline]
    fn mul(&self, other: &$T) -> $T { *self * *other }
}

#[cfg(not(test))]
impl Div<$T,$T> for $T {
    /// Integer division, truncated towards 0.
    ///
    /// # Examples
    ///
    /// ~~~
    /// assert!( 8 /  3 ==  2);
    /// assert!( 8 / -3 == -2);
    /// assert!(-8 /  3 == -2);
    /// assert!(-8 / -3 ==  2);
    ///
    /// assert!( 1 /  2 ==  0);
    /// assert!( 1 / -2 ==  0);
    /// assert!(-1 /  2 ==  0);
    /// assert!(-1 / -2 ==  0);
    /// ~~~
    #[inline]
    fn div(&self, other: &$T) -> $T { *self / *other }
}

#[cfg(not(test))]
impl Rem<$T,$T> for $T {
    /// Returns the integer remainder after division, satisfying:
    ///
    /// ~~~
    /// # let n = 1;
    /// # let d = 2;
    /// assert!((n / d) * d + (n % d) == n)
    /// ~~~
    ///
    /// # Examples
    ///
    /// ~~~
    /// assert!( 8 %  3 ==  2);
    /// assert!( 8 % -3 ==  2);
    /// assert!(-8 %  3 == -2);
    /// assert!(-8 % -3 == -2);
    ///
    /// assert!( 1 %  2 ==  1);
    /// assert!( 1 % -2 ==  1);
    /// assert!(-1 %  2 == -1);
    /// assert!(-1 % -2 == -1);
    /// ~~~
    #[inline]
    fn rem(&self, other: &$T) -> $T { *self % *other }
}

#[cfg(not(test))]
impl Neg<$T> for $T {
    #[inline]
    fn neg(&self) -> $T { -*self }
}

impl Signed for $T {
    /// Computes the absolute value
    #[inline]
    fn abs(&self) -> $T {
        if self.is_negative() { -*self } else { *self }
    }

    ///
    /// The positive difference of two numbers. Returns `0` if the number is less than or
    /// equal to `other`, otherwise the difference between`self` and `other` is returned.
    ///
    #[inline]
    fn abs_sub(&self, other: &$T) -> $T {
        if *self <= *other { 0 } else { *self - *other }
    }

    ///
    /// # Returns
    ///
    /// - `0` if the number is zero
    /// - `1` if the number is positive
    /// - `-1` if the number is negative
    ///
    #[inline]
    fn signum(&self) -> $T {
        match *self {
            n if n > 0 =>  1,
            0          =>  0,
            _          => -1,
        }
    }

    /// Returns true if the number is positive
    #[inline]
    fn is_positive(&self) -> bool { *self > 0 }

    /// Returns true if the number is negative
    #[inline]
    fn is_negative(&self) -> bool { *self < 0 }
}

#[cfg(not(test))]
impl BitOr<$T,$T> for $T {
    #[inline]
    fn bitor(&self, other: &$T) -> $T { *self | *other }
}

#[cfg(not(test))]
impl BitAnd<$T,$T> for $T {
    #[inline]
    fn bitand(&self, other: &$T) -> $T { *self & *other }
}

#[cfg(not(test))]
impl BitXor<$T,$T> for $T {
    #[inline]
    fn bitxor(&self, other: &$T) -> $T { *self ^ *other }
}

#[cfg(not(test))]
impl Shl<$T,$T> for $T {
    #[inline]
    fn shl(&self, other: &$T) -> $T { *self << *other }
}

#[cfg(not(test))]
impl Shr<$T,$T> for $T {
    #[inline]
    fn shr(&self, other: &$T) -> $T { *self >> *other }
}

#[cfg(not(test))]
impl Not<$T> for $T {
    #[inline]
    fn not(&self) -> $T { !*self }
}

impl Bounded for $T {
    #[inline]
    fn min_value() -> $T { MIN }

    #[inline]
    fn max_value() -> $T { MAX }
}

impl Int for $T {}

impl Primitive for $T {}

// String conversion functions and impl str -> num

/// Parse a byte slice as a number in the given base.
///
/// Yields an `Option` because `buf` may or may not actually be parseable.
///
/// # Examples
///
/// ```rust
/// let digits = [49,50,51,52,53,54,55,56,57];
/// let base   = 10;
/// let num    = std::i64::parse_bytes(digits, base);
/// ```
#[inline]
pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<$T> {
    strconv::from_str_bytes_common(buf, radix, true, false, false,
                               strconv::ExpNone, false, false)
}

impl FromStr for $T {
    #[inline]
    fn from_str(s: &str) -> Option<$T> {
        strconv::from_str_common(s, 10u, true, false, false,
                             strconv::ExpNone, false, false)
    }
}

impl FromStrRadix for $T {
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<$T> {
        strconv::from_str_common(s, radix, true, false, false,
                             strconv::ExpNone, false, false)
    }
}

// String conversion functions and impl num -> str

/// Convert to a string as a byte slice in a given base.
#[inline]
pub fn to_str_bytes<U>(n: $T, radix: uint, f: |v: &[u8]| -> U) -> U {
    // The radix can be as low as 2, so we need at least 64 characters for a
    // base 2 number, and then we need another for a possible '-' character.
    let mut buf = [0u8, ..65];
    let mut cur = 0;
    strconv::int_to_str_bytes_common(n, radix, strconv::SignNeg, |i| {
        buf[cur] = i;
        cur += 1;
    });
    f(buf.slice(0, cur))
}

impl ToStrRadix for $T {
    /// Convert to a string in a given base.
    #[inline]
    fn to_str_radix(&self, radix: uint) -> ~str {
        let mut buf = Vec::new();
        strconv::int_to_str_bytes_common(*self, radix, strconv::SignNeg, |i| {
            buf.push(i);
        });
        // We know we generated valid utf-8, so we don't need to go through that
        // check.
        unsafe { str::raw::from_utf8_owned(buf.move_iter().collect()) }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    use int;
    use i32;
    use num;
    use num::Bitwise;
    use num::CheckedDiv;
    use num::ToStrRadix;
    use str::StrSlice;

    #[test]
    fn test_overflows() {
        assert!(MAX > 0);
        assert!(MIN <= 0);
        assert_eq!(MIN + MAX + 1, 0);
    }

    #[test]
    fn test_num() {
        num::test_num(10 as $T, 2 as $T);
    }

    #[test]
    pub fn test_abs() {
        assert_eq!((1 as $T).abs(), 1 as $T);
        assert_eq!((0 as $T).abs(), 0 as $T);
        assert_eq!((-1 as $T).abs(), 1 as $T);
    }

    #[test]
    fn test_abs_sub() {
        assert_eq!((-1 as $T).abs_sub(&(1 as $T)), 0 as $T);
        assert_eq!((1 as $T).abs_sub(&(1 as $T)), 0 as $T);
        assert_eq!((1 as $T).abs_sub(&(0 as $T)), 1 as $T);
        assert_eq!((1 as $T).abs_sub(&(-1 as $T)), 2 as $T);
    }

    #[test]
    fn test_signum() {
        assert_eq!((1 as $T).signum(), 1 as $T);
        assert_eq!((0 as $T).signum(), 0 as $T);
        assert_eq!((-0 as $T).signum(), 0 as $T);
        assert_eq!((-1 as $T).signum(), -1 as $T);
    }

    #[test]
    fn test_is_positive() {
        assert!((1 as $T).is_positive());
        assert!(!(0 as $T).is_positive());
        assert!(!(-0 as $T).is_positive());
        assert!(!(-1 as $T).is_positive());
    }

    #[test]
    fn test_is_negative() {
        assert!(!(1 as $T).is_negative());
        assert!(!(0 as $T).is_negative());
        assert!(!(-0 as $T).is_negative());
        assert!((-1 as $T).is_negative());
    }

    #[test]
    fn test_bitwise() {
        assert_eq!(0b1110 as $T, (0b1100 as $T).bitor(&(0b1010 as $T)));
        assert_eq!(0b1000 as $T, (0b1100 as $T).bitand(&(0b1010 as $T)));
        assert_eq!(0b0110 as $T, (0b1100 as $T).bitxor(&(0b1010 as $T)));
        assert_eq!(0b1110 as $T, (0b0111 as $T).shl(&(1 as $T)));
        assert_eq!(0b0111 as $T, (0b1110 as $T).shr(&(1 as $T)));
        assert_eq!(-(0b11 as $T) - (1 as $T), (0b11 as $T).not());
    }

    #[test]
    fn test_count_ones() {
        assert_eq!((0b0101100 as $T).count_ones(), 3);
        assert_eq!((0b0100001 as $T).count_ones(), 2);
        assert_eq!((0b1111001 as $T).count_ones(), 5);
    }

    #[test]
    fn test_count_zeros() {
        assert_eq!((0b0101100 as $T).count_zeros(), BITS as $T - 3);
        assert_eq!((0b0100001 as $T).count_zeros(), BITS as $T - 2);
        assert_eq!((0b1111001 as $T).count_zeros(), BITS as $T - 5);
    }

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
    fn test_to_str() {
        assert_eq!((0 as $T).to_str_radix(10u), "0".to_owned());
        assert_eq!((1 as $T).to_str_radix(10u), "1".to_owned());
        assert_eq!((-1 as $T).to_str_radix(10u), "-1".to_owned());
        assert_eq!((127 as $T).to_str_radix(16u), "7f".to_owned());
        assert_eq!((100 as $T).to_str_radix(10u), "100".to_owned());

    }

    #[test]
    fn test_int_to_str_overflow() {
        let mut i8_val: i8 = 127_i8;
        assert_eq!(i8_val.to_str(), "127".to_owned());

        i8_val += 1 as i8;
        assert_eq!(i8_val.to_str(), "-128".to_owned());

        let mut i16_val: i16 = 32_767_i16;
        assert_eq!(i16_val.to_str(), "32767".to_owned());

        i16_val += 1 as i16;
        assert_eq!(i16_val.to_str(), "-32768".to_owned());

        let mut i32_val: i32 = 2_147_483_647_i32;
        assert_eq!(i32_val.to_str(), "2147483647".to_owned());

        i32_val += 1 as i32;
        assert_eq!(i32_val.to_str(), "-2147483648".to_owned());

        let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
        assert_eq!(i64_val.to_str(), "9223372036854775807".to_owned());

        i64_val += 1 as i64;
        assert_eq!(i64_val.to_str(), "-9223372036854775808".to_owned());
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

    #[test]
    fn test_signed_checked_div() {
        assert_eq!(10i.checked_div(&2), Some(5));
        assert_eq!(5i.checked_div(&0), None);
        assert_eq!(int::MIN.checked_div(&-1), None);
    }
}

))
