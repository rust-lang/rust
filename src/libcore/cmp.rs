// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The `Ord` and `Eq` comparison traits

This module contains the definition of both `Ord` and `Eq` which define
the common interfaces for doing comparison. Both are language items
that the compiler uses to implement the comparison operators. Rust code
may implement `Ord` to overload the `<`, `<=`, `>`, and `>=` operators,
and `Eq` to overload the `==` and `!=` operators.

*/

/**
* Trait for values that can be compared for equality and inequality.
*
* This trait allows partial equality, where types can be unordered instead of strictly equal or
* unequal. For example, with the built-in floating-point types `a == b` and `a != b` will both
* evaluate to false if either `a` or `b` is NaN (cf. IEEE 754-2008 section 5.11).
*
* Eventually, this will be implemented by default for types that implement `TotalEq`.
*/
#[lang="eq"]
pub trait Eq {
    fn eq(&self, other: &Self) -> bool;
    fn ne(&self, other: &Self) -> bool;
}

/// Trait for equality comparisons where `a == b` and `a != b` are strict inverses.
pub trait TotalEq {
    fn equals(&self, other: &Self) -> bool;
}

macro_rules! totaleq_impl(
    ($t:ty) => {
        impl TotalEq for $t {
            #[inline(always)]
            fn equals(&self, other: &$t) -> bool { *self == *other }
        }
    }
)

totaleq_impl!(bool)

totaleq_impl!(u8)
totaleq_impl!(u16)
totaleq_impl!(u32)
totaleq_impl!(u64)

totaleq_impl!(i8)
totaleq_impl!(i16)
totaleq_impl!(i32)
totaleq_impl!(i64)

totaleq_impl!(int)
totaleq_impl!(uint)

totaleq_impl!(char)

/// Trait for testing approximate equality
pub trait ApproxEq<Eps> {
    fn approx_epsilon() -> Eps;
    fn approx_eq(&self, other: &Self) -> bool;
    fn approx_eq_eps(&self, other: &Self, approx_epsilon: &Eps) -> bool;
}

#[deriving(Clone, Eq)]
pub enum Ordering { Less = -1, Equal = 0, Greater = 1 }

/// Trait for types that form a total order
pub trait TotalOrd: TotalEq {
    fn cmp(&self, other: &Self) -> Ordering;
}

impl TotalOrd for Ordering {
    #[inline(always)]
    fn cmp(&self, other: &Ordering) -> Ordering {
        (*self as int).cmp(&(*other as int))
    }
}

impl Ord for Ordering {
    #[inline(always)]
    fn lt(&self, other: &Ordering) -> bool { (*self as int) < (*other as int) }
    #[inline(always)]
    fn le(&self, other: &Ordering) -> bool { (*self as int) <= (*other as int) }
    #[inline(always)]
    fn gt(&self, other: &Ordering) -> bool { (*self as int) > (*other as int) }
    #[inline(always)]
    fn ge(&self, other: &Ordering) -> bool { (*self as int) >= (*other as int) }
}

macro_rules! totalord_impl(
    ($t:ty) => {
        impl TotalOrd for $t {
            #[inline(always)]
            fn cmp(&self, other: &$t) -> Ordering {
                if *self < *other { Less }
                else if *self > *other { Greater }
                else { Equal }
            }
        }
    }
)

totalord_impl!(u8)
totalord_impl!(u16)
totalord_impl!(u32)
totalord_impl!(u64)

totalord_impl!(i8)
totalord_impl!(i16)
totalord_impl!(i32)
totalord_impl!(i64)

totalord_impl!(int)
totalord_impl!(uint)

totalord_impl!(char)

pub fn cmp2<A:TotalOrd,B:TotalOrd>(
    a1: &A, b1: &B,
    a2: &A, b2: &B) -> Ordering
{
    //! Compares (a1, b1) against (a2, b2), where the a values are more significant.

    match a1.cmp(a2) {
        Less => Less,
        Greater => Greater,
        Equal => b1.cmp(b2)
    }
}

/**
Return `o1` if it is not `Equal`, otherwise `o2`. Simulates the
lexical ordering on a type `(int, int)`.
*/
// used in deriving code in libsyntax
#[inline(always)]
pub fn lexical_ordering(o1: Ordering, o2: Ordering) -> Ordering {
    match o1 {
        Equal => o2,
        _ => o1
    }
}

/**
* Trait for values that can be compared for a sort-order.
*
* Eventually this may be simplified to only require
* an `le` method, with the others generated from
* default implementations. However it should remain
* possible to implement the others separately, for
* compatibility with floating-point NaN semantics
* (cf. IEEE 754-2008 section 5.11).
*/
#[lang="ord"]
pub trait Ord {
    fn lt(&self, other: &Self) -> bool;
    fn le(&self, other: &Self) -> bool;
    fn ge(&self, other: &Self) -> bool;
    fn gt(&self, other: &Self) -> bool;
}

#[inline(always)]
pub fn lt<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2)
}

#[inline(always)]
pub fn le<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).le(v2)
}

#[inline(always)]
pub fn eq<T:Eq>(v1: &T, v2: &T) -> bool {
    (*v1).eq(v2)
}

#[inline(always)]
pub fn ne<T:Eq>(v1: &T, v2: &T) -> bool {
    (*v1).ne(v2)
}

#[inline(always)]
pub fn ge<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).ge(v2)
}

#[inline(always)]
pub fn gt<T:Ord>(v1: &T, v2: &T) -> bool {
    (*v1).gt(v2)
}

/// The equivalence relation. Two values may be equivalent even if they are
/// of different types. The most common use case for this relation is
/// container types; e.g. it is often desirable to be able to use `&str`
/// values to look up entries in a container with `~str` keys.
pub trait Equiv<T> {
    fn equiv(&self, other: &T) -> bool;
}

#[inline(always)]
pub fn min<T:Ord>(v1: T, v2: T) -> T {
    if v1 < v2 { v1 } else { v2 }
}

#[inline(always)]
pub fn max<T:Ord>(v1: T, v2: T) -> T {
    if v1 > v2 { v1 } else { v2 }
}

#[cfg(test)]
mod test {
    use super::lexical_ordering;

    #[test]
    fn test_int_totalord() {
        assert_eq!(5.cmp(&10), Less);
        assert_eq!(10.cmp(&5), Greater);
        assert_eq!(5.cmp(&5), Equal);
        assert_eq!((-5).cmp(&12), Less);
        assert_eq!(12.cmp(-5), Greater);
    }

    #[test]
    fn test_cmp2() {
        assert_eq!(cmp2(1, 2, 3, 4), Less);
        assert_eq!(cmp2(3, 2, 3, 4), Less);
        assert_eq!(cmp2(5, 2, 3, 4), Greater);
        assert_eq!(cmp2(5, 5, 5, 4), Greater);
    }

    #[test]
    fn test_int_totaleq() {
        assert!(5.equals(&5));
        assert!(!2.equals(&17));
    }

    #[test]
    fn test_ordering_order() {
        assert!(Less < Equal);
        assert_eq!(Greater.cmp(&Less), Greater);
    }

    #[test]
    fn test_lexical_ordering() {
        fn t(o1: Ordering, o2: Ordering, e: Ordering) {
            assert_eq!(lexical_ordering(o1, o2), e);
        }
        for [Less, Equal, Greater].each |&o| {
            t(Less, o, Less);
            t(Equal, o, o);
            t(Greater, o, Greater);
         }
    }
}
