// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Defines the `PartialOrd` and `PartialEq` comparison traits.
//!
//! This module defines both `PartialOrd` and `PartialEq` traits which are used by the
//! compiler to implement comparison operators. Rust programs may implement
//!`PartialOrd` to overload the `<`, `<=`, `>`, and `>=` operators, and may implement
//! `PartialEq` to overload the `==` and `!=` operators.
//!
//! For example, to define a type with a customized definition for the PartialEq
//! operators, you could do the following:
//!
//! ```rust
//! // Our type.
//! struct SketchyNum {
//!     num : int
//! }
//!
//! // Our implementation of `PartialEq` to support `==` and `!=`.
//! impl PartialEq for SketchyNum {
//!     // Our custom eq allows numbers which are near each other to be equal! :D
//!     fn eq(&self, other: &SketchyNum) -> bool {
//!         (self.num - other.num).abs() < 5
//!     }
//! }
//!
//! // Now these binary operators will work when applied!
//! assert!(SketchyNum {num: 37} == SketchyNum {num: 34});
//! assert!(SketchyNum {num: 25} != SketchyNum {num: 57});
//! ```

/// Trait for values that can be compared for equality and inequality.
///
/// This trait allows partial equality, where types can be unordered instead of
/// strictly equal or unequal. For example, with the built-in floating-point
/// types `a == b` and `a != b` will both evaluate to false if either `a` or
/// `b` is NaN (cf. IEEE 754-2008 section 5.11).
///
/// PartialEq only requires the `eq` method to be implemented; `ne` is its negation by
/// default.
///
/// Eventually, this will be implemented by default for types that implement
/// `Eq`.
#[lang="eq"]
pub trait PartialEq {
    /// This method tests for `self` and `other` values to be equal, and is used by `==`.
    fn eq(&self, other: &Self) -> bool;

    /// This method tests for `!=`.
    #[inline]
    fn ne(&self, other: &Self) -> bool { !self.eq(other) }
}

/// Trait for equality comparisons which are [equivalence relations](
/// https://en.wikipedia.org/wiki/Equivalence_relation).
///
/// This means, that in addition to `a == b` and `a != b` being strict
/// inverses, the equality must be (for all `a`, `b` and `c`):
///
/// - reflexive: `a == a`;
/// - symmetric: `a == b` implies `b == a`; and
/// - transitive: `a == b` and `b == c` implies `a == c`.
pub trait Eq: PartialEq {
    // FIXME #13101: this method is used solely by #[deriving] to
    // assert that every component of a type implements #[deriving]
    // itself, the current deriving infrastructure means doing this
    // assertion without using a method on this trait is nearly
    // impossible.
    //
    // This should never be implemented by hand.
    #[doc(hidden)]
    #[inline(always)]
    fn assert_receiver_is_total_eq(&self) {}
}

/// An ordering is, e.g, a result of a comparison between two values.
#[deriving(Clone, PartialEq, Show)]
pub enum Ordering {
   /// An ordering where a compared value is less [than another].
   Less = -1,
   /// An ordering where a compared value is equal [to another].
   Equal = 0,
   /// An ordering where a compared value is greater [than another].
   Greater = 1
}

/// Trait for types that form a [total order](
/// https://en.wikipedia.org/wiki/Total_order).
///
/// An order is a total order if it is (for all `a`, `b` and `c`):
///
/// - total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is
///   true; and
/// - transitive, `a < b` and `b < c` implies `a < c`. The same must hold for
///   both `==` and `>`.
pub trait Ord: Eq + PartialOrd {
    /// This method returns an ordering between `self` and `other` values.
    ///
    /// By convention, `self.cmp(&other)` returns the ordering matching
    /// the expression `self <operator> other` if true.  For example:
    ///
    /// ```
    /// assert_eq!( 5u.cmp(&10), Less);     // because 5 < 10
    /// assert_eq!(10u.cmp(&5),  Greater);  // because 10 > 5
    /// assert_eq!( 5u.cmp(&5),  Equal);    // because 5 == 5
    /// ```
    fn cmp(&self, other: &Self) -> Ordering;
}

impl Eq for Ordering {}

impl Ord for Ordering {
    #[inline]
    fn cmp(&self, other: &Ordering) -> Ordering {
        (*self as int).cmp(&(*other as int))
    }
}

impl PartialOrd for Ordering {
    #[inline]
    fn lt(&self, other: &Ordering) -> bool { (*self as int) < (*other as int) }
}

/// Combine orderings, lexically.
///
/// For example for a type `(int, int)`, two comparisons could be done.
/// If the first ordering is different, the first ordering is all that must be returned.
/// If the first ordering is equal, then second ordering is returned.
#[inline]
pub fn lexical_ordering(o1: Ordering, o2: Ordering) -> Ordering {
    match o1 {
        Equal => o2,
        _ => o1
    }
}

/// Trait for values that can be compared for a sort-order.
///
/// PartialOrd only requires implementation of the `lt` method,
/// with the others generated from default implementations.
///
/// However it remains possible to implement the others separately,
/// for compatibility with floating-point NaN semantics
/// (cf. IEEE 754-2008 section 5.11).
#[lang="ord"]
pub trait PartialOrd: PartialEq {
    /// This method tests less than (for `self` and `other`) and is used by the `<` operator.
    fn lt(&self, other: &Self) -> bool;

    /// This method tests less than or equal to (`<=`).
    #[inline]
    fn le(&self, other: &Self) -> bool { !other.lt(self) }

    /// This method tests greater than (`>`).
    #[inline]
    fn gt(&self, other: &Self) -> bool {  other.lt(self) }

    /// This method tests greater than or equal to (`>=`).
    #[inline]
    fn ge(&self, other: &Self) -> bool { !self.lt(other) }
}

/// The equivalence relation. Two values may be equivalent even if they are
/// of different types. The most common use case for this relation is
/// container types; e.g. it is often desirable to be able to use `&str`
/// values to look up entries in a container with `String` keys.
pub trait Equiv<T> {
    /// Implement this function to decide equivalent values.
    fn equiv(&self, other: &T) -> bool;
}

/// Compare and return the minimum of two values.
#[inline]
pub fn min<T: Ord>(v1: T, v2: T) -> T {
    if v1 < v2 { v1 } else { v2 }
}

/// Compare and return the maximum of two values.
#[inline]
pub fn max<T: Ord>(v1: T, v2: T) -> T {
    if v1 > v2 { v1 } else { v2 }
}

// Implementation of PartialEq, Eq, PartialOrd and Ord for primitive types
#[cfg(not(test))]
mod impls {
    use cmp::{PartialOrd, Ord, PartialEq, Eq, Ordering,
              Less, Greater, Equal};

    macro_rules! eq_impl(
        ($($t:ty)*) => ($(
            impl PartialEq for $t {
                #[inline]
                fn eq(&self, other: &$t) -> bool { (*self) == (*other) }
                #[inline]
                fn ne(&self, other: &$t) -> bool { (*self) != (*other) }
            }
        )*)
    )

    impl PartialEq for () {
        #[inline]
        fn eq(&self, _other: &()) -> bool { true }
        #[inline]
        fn ne(&self, _other: &()) -> bool { false }
    }

    eq_impl!(bool char uint u8 u16 u32 u64 int i8 i16 i32 i64 f32 f64)

    macro_rules! totaleq_impl(
        ($($t:ty)*) => ($(
            impl Eq for $t {}
        )*)
    )

    totaleq_impl!(() bool char uint u8 u16 u32 u64 int i8 i16 i32 i64)

    macro_rules! ord_impl(
        ($($t:ty)*) => ($(
            impl PartialOrd for $t {
                #[inline]
                fn lt(&self, other: &$t) -> bool { (*self) < (*other) }
                #[inline]
                fn le(&self, other: &$t) -> bool { (*self) <= (*other) }
                #[inline]
                fn ge(&self, other: &$t) -> bool { (*self) >= (*other) }
                #[inline]
                fn gt(&self, other: &$t) -> bool { (*self) > (*other) }
            }
        )*)
    )

    impl PartialOrd for () {
        #[inline]
        fn lt(&self, _other: &()) -> bool { false }
    }

    impl PartialOrd for bool {
        #[inline]
        fn lt(&self, other: &bool) -> bool {
            (*self as u8) < (*other as u8)
        }
    }

    ord_impl!(char uint u8 u16 u32 u64 int i8 i16 i32 i64 f32 f64)

    macro_rules! totalord_impl(
        ($($t:ty)*) => ($(
            impl Ord for $t {
                #[inline]
                fn cmp(&self, other: &$t) -> Ordering {
                    if *self < *other { Less }
                    else if *self > *other { Greater }
                    else { Equal }
                }
            }
        )*)
    )

    impl Ord for () {
        #[inline]
        fn cmp(&self, _other: &()) -> Ordering { Equal }
    }

    impl Ord for bool {
        #[inline]
        fn cmp(&self, other: &bool) -> Ordering {
            (*self as u8).cmp(&(*other as u8))
        }
    }

    totalord_impl!(char uint u8 u16 u32 u64 int i8 i16 i32 i64)

    // & pointers
    impl<'a, T: PartialEq> PartialEq for &'a T {
        #[inline]
        fn eq(&self, other: & &'a T) -> bool { *(*self) == *(*other) }
        #[inline]
        fn ne(&self, other: & &'a T) -> bool { *(*self) != *(*other) }
    }
    impl<'a, T: PartialOrd> PartialOrd for &'a T {
        #[inline]
        fn lt(&self, other: & &'a T) -> bool { *(*self) < *(*other) }
        #[inline]
        fn le(&self, other: & &'a T) -> bool { *(*self) <= *(*other) }
        #[inline]
        fn ge(&self, other: & &'a T) -> bool { *(*self) >= *(*other) }
        #[inline]
        fn gt(&self, other: & &'a T) -> bool { *(*self) > *(*other) }
    }
    impl<'a, T: Ord> Ord for &'a T {
        #[inline]
        fn cmp(&self, other: & &'a T) -> Ordering { (**self).cmp(*other) }
    }
    impl<'a, T: Eq> Eq for &'a T {}

    // &mut pointers
    impl<'a, T: PartialEq> PartialEq for &'a mut T {
        #[inline]
        fn eq(&self, other: &&'a mut T) -> bool { **self == *(*other) }
        #[inline]
        fn ne(&self, other: &&'a mut T) -> bool { **self != *(*other) }
    }
    impl<'a, T: PartialOrd> PartialOrd for &'a mut T {
        #[inline]
        fn lt(&self, other: &&'a mut T) -> bool { **self < **other }
        #[inline]
        fn le(&self, other: &&'a mut T) -> bool { **self <= **other }
        #[inline]
        fn ge(&self, other: &&'a mut T) -> bool { **self >= **other }
        #[inline]
        fn gt(&self, other: &&'a mut T) -> bool { **self > **other }
    }
    impl<'a, T: Ord> Ord for &'a mut T {
        #[inline]
        fn cmp(&self, other: &&'a mut T) -> Ordering { (**self).cmp(*other) }
    }
    impl<'a, T: Eq> Eq for &'a mut T {}

    // @ pointers
    impl<T:PartialEq> PartialEq for @T {
        #[inline]
        fn eq(&self, other: &@T) -> bool { *(*self) == *(*other) }
        #[inline]
        fn ne(&self, other: &@T) -> bool { *(*self) != *(*other) }
    }
    impl<T:PartialOrd> PartialOrd for @T {
        #[inline]
        fn lt(&self, other: &@T) -> bool { *(*self) < *(*other) }
        #[inline]
        fn le(&self, other: &@T) -> bool { *(*self) <= *(*other) }
        #[inline]
        fn ge(&self, other: &@T) -> bool { *(*self) >= *(*other) }
        #[inline]
        fn gt(&self, other: &@T) -> bool { *(*self) > *(*other) }
    }
    impl<T: Ord> Ord for @T {
        #[inline]
        fn cmp(&self, other: &@T) -> Ordering { (**self).cmp(*other) }
    }
    impl<T: Eq> Eq for @T {}
}

#[cfg(test)]
mod test {
    use super::lexical_ordering;

    #[test]
    fn test_int_totalord() {
        assert_eq!(5u.cmp(&10), Less);
        assert_eq!(10u.cmp(&5), Greater);
        assert_eq!(5u.cmp(&5), Equal);
        assert_eq!((-5u).cmp(&12), Less);
        assert_eq!(12u.cmp(-5), Greater);
    }

    #[test]
    fn test_mut_int_totalord() {
        assert_eq!((&mut 5u).cmp(&10), Less);
        assert_eq!((&mut 10u).cmp(&5), Greater);
        assert_eq!((&mut 5u).cmp(&5), Equal);
        assert_eq!((&mut -5u).cmp(&12), Less);
        assert_eq!((&mut 12u).cmp(-5), Greater);
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

        let xs = [Less, Equal, Greater];
        for &o in xs.iter() {
            t(Less, o, Less);
            t(Equal, o, o);
            t(Greater, o, Greater);
         }
    }

    #[test]
    fn test_user_defined_eq() {
        // Our type.
        struct SketchyNum {
            num : int
        }

        // Our implementation of `PartialEq` to support `==` and `!=`.
        impl PartialEq for SketchyNum {
            // Our custom eq allows numbers which are near each other to be equal! :D
            fn eq(&self, other: &SketchyNum) -> bool {
                (self.num - other.num).abs() < 5
            }
        }

        // Now these binary operators will work when applied!
        assert!(SketchyNum {num: 37} == SketchyNum {num: 34});
        assert!(SketchyNum {num: 25} != SketchyNum {num: 57});
    }
}
