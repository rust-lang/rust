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

macro_rules! uint_module (($T:ty, $T_SIGNED:ty, $bits:expr) => (

pub static BITS : uint = $bits;
pub static BYTES : uint = ($bits / 8);

pub static MIN: $T = 0 as $T;
pub static MAX: $T = 0 as $T - 1 as $T;

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
    #[inline]
    fn div(&self, other: &$T) -> $T { *self / *other }
}

#[cfg(not(test))]
impl Rem<$T,$T> for $T {
    #[inline]
    fn rem(&self, other: &$T) -> $T { *self % *other }
}

#[cfg(not(test))]
impl Neg<$T> for $T {
    #[inline]
    fn neg(&self) -> $T { -(*self as $T_SIGNED) as $T }
}

impl Unsigned for $T {}

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

impl Bitwise for $T {
    /// Returns the number of ones in the binary representation of the number.
    #[inline]
    fn count_ones(&self) -> $T {
        (*self as $T_SIGNED).count_ones() as $T
    }

    /// Returns the number of leading zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn leading_zeros(&self) -> $T {
        (*self as $T_SIGNED).leading_zeros() as $T
    }

    /// Returns the number of trailing zeros in the in the binary representation
    /// of the number.
    #[inline]
    fn trailing_zeros(&self) -> $T {
        (*self as $T_SIGNED).trailing_zeros() as $T
    }
}

impl CheckedDiv for $T {
    #[inline]
    fn checked_div(&self, v: &$T) -> Option<$T> {
        if *v == 0 {
            None
        } else {
            Some(self / *v)
        }
    }
}

impl Int for $T {}

impl Primitive for $T {}

impl Default for $T {
    #[inline]
    fn default() -> $T { 0 }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    use num;
    use num::CheckedDiv;
    use num::Bitwise;

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
    fn test_bitwise() {
        assert!(0b1110 as $T == (0b1100 as $T).bitor(&(0b1010 as $T)));
        assert!(0b1000 as $T == (0b1100 as $T).bitand(&(0b1010 as $T)));
        assert!(0b0110 as $T == (0b1100 as $T).bitxor(&(0b1010 as $T)));
        assert!(0b1110 as $T == (0b0111 as $T).shl(&(1 as $T)));
        assert!(0b0111 as $T == (0b1110 as $T).shr(&(1 as $T)));
        assert!(MAX - (0b1011 as $T) == (0b1011 as $T).not());
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
    fn test_unsigned_checked_div() {
        assert!(10u.checked_div(&2) == Some(5));
        assert!(5u.checked_div(&0) == None);
    }
}

))
