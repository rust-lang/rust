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

#[cfg(not(test))]
impl Ord for $T {
    #[inline]
    fn lt(&self, other: &$T) -> bool { *self < *other }
}
#[cfg(not(test))]
impl TotalEq for $T {}
#[cfg(not(test))]
impl Eq for $T {
    #[inline]
    fn eq(&self, other: &$T) -> bool { *self == *other }
}
#[cfg(not(test))]
impl TotalOrd for $T {
    #[inline]
    fn cmp(&self, other: &$T) -> Ordering {
        if *self < *other { Less }
        else if *self > *other { Greater }
        else { Equal }
    }
}

impl Num for $T {}

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

impl Default for $T {
    #[inline]
    fn default() -> $T { 0 }
}

impl Int for $T {}

impl Primitive for $T {}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    use int;
    use num;
    use num::Bitwise;
    use num::CheckedDiv;

    #[test]
    fn test_overflows() {
        assert!(MAX > 0);
        assert!(MIN <= 0);
        assert!(MIN + MAX + 1 == 0);
    }

    #[test]
    fn test_num() {
        num::test_num(10 as $T, 2 as $T);
    }

    #[test]
    pub fn test_abs() {
        assert!((1 as $T).abs() == 1 as $T);
        assert!((0 as $T).abs() == 0 as $T);
        assert!((-1 as $T).abs() == 1 as $T);
    }

    #[test]
    fn test_abs_sub() {
        assert!((-1 as $T).abs_sub(&(1 as $T)) == 0 as $T);
        assert!((1 as $T).abs_sub(&(1 as $T)) == 0 as $T);
        assert!((1 as $T).abs_sub(&(0 as $T)) == 1 as $T);
        assert!((1 as $T).abs_sub(&(-1 as $T)) == 2 as $T);
    }

    #[test]
    fn test_signum() {
        assert!((1 as $T).signum() == 1 as $T);
        assert!((0 as $T).signum() == 0 as $T);
        assert!((-0 as $T).signum() == 0 as $T);
        assert!((-1 as $T).signum() == -1 as $T);
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
        assert!(0b1110 as $T == (0b1100 as $T).bitor(&(0b1010 as $T)));
        assert!(0b1000 as $T == (0b1100 as $T).bitand(&(0b1010 as $T)));
        assert!(0b0110 as $T == (0b1100 as $T).bitxor(&(0b1010 as $T)));
        assert!(0b1110 as $T == (0b0111 as $T).shl(&(1 as $T)));
        assert!(0b0111 as $T == (0b1110 as $T).shr(&(1 as $T)));
        assert!(-(0b11 as $T) - (1 as $T) == (0b11 as $T).not());
    }

    #[test]
    fn test_count_ones() {
        assert!((0b0101100 as $T).count_ones() == 3);
        assert!((0b0100001 as $T).count_ones() == 2);
        assert!((0b1111001 as $T).count_ones() == 5);
    }

    #[test]
    fn test_count_zeros() {
        assert!((0b0101100 as $T).count_zeros() == BITS as $T - 3);
        assert!((0b0100001 as $T).count_zeros() == BITS as $T - 2);
        assert!((0b1111001 as $T).count_zeros() == BITS as $T - 5);
    }

    #[test]
    fn test_signed_checked_div() {
        assert!(10i.checked_div(&2) == Some(5));
        assert!(5i.checked_div(&0) == None);
        assert!(int::MIN.checked_div(&-1) == None);
    }
}

))
