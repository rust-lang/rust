// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functionality for ordering and comparison.
//!
//! This module defines both `PartialOrd` and `PartialEq` traits which are used
//! by the compiler to implement comparison operators. Rust programs may
//! implement `PartialOrd` to overload the `<`, `<=`, `>`, and `>=` operators,
//! and may implement `PartialEq` to overload the `==` and `!=` operators.

#![stable(feature = "rust1", since = "1.0.0")]

use self::Ordering::*;

use mem;
use marker::Sized;
use option::Option::{self, Some};

/// Trait for equality comparisons which are [partial equivalence
/// relations](http://en.wikipedia.org/wiki/Partial_equivalence_relation).
///
/// This trait allows for partial equality, for types that do not have a full
/// equivalence relation.  For example, in floating point numbers `NaN != NaN`,
/// so floating point types implement `PartialEq` but not `Eq`.
///
/// Formally, the equality must be (for all `a`, `b` and `c`):
///
/// - symmetric: `a == b` implies `b == a`; and
/// - transitive: `a == b` and `b == c` implies `a == c`.
///
/// Note that these requirements mean that the trait itself must be implemented
/// symmetrically and transitively: if `T: PartialEq<U>` and `U: PartialEq<V>`
/// then `U: PartialEq<T>` and `T: PartialEq<V>`.
///
/// PartialEq only requires the `eq` method to be implemented; `ne` is defined
/// in terms of it by default. Any manual implementation of `ne` *must* respect
/// the rule that `eq` is a strict inverse of `ne`; that is, `!(a == b)` if and
/// only if `a != b`.
#[lang = "eq"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait PartialEq<Rhs: ?Sized = Self> {
    /// This method tests for `self` and `other` values to be equal, and is used
    /// by `==`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn eq(&self, other: &Rhs) -> bool;

    /// This method tests for `!=`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ne(&self, other: &Rhs) -> bool { !self.eq(other) }
}

/// Trait for equality comparisons which are [equivalence relations](
/// https://en.wikipedia.org/wiki/Equivalence_relation).
///
/// This means, that in addition to `a == b` and `a != b` being strict inverses, the equality must
/// be (for all `a`, `b` and `c`):
///
/// - reflexive: `a == a`;
/// - symmetric: `a == b` implies `b == a`; and
/// - transitive: `a == b` and `b == c` implies `a == c`.
///
/// This property cannot be checked by the compiler, and therefore `Eq` implies
/// `PartialEq`, and has no extra methods.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Eq: PartialEq<Self> {
    // FIXME #13101: this method is used solely by #[deriving] to
    // assert that every component of a type implements #[deriving]
    // itself, the current deriving infrastructure means doing this
    // assertion without using a method on this trait is nearly
    // impossible.
    //
    // This should never be implemented by hand.
    #[doc(hidden)]
    #[inline(always)]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn assert_receiver_is_total_eq(&self) {}
}

/// An `Ordering` is the result of a comparison between two values.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
///
/// let result = 1.cmp(&2);
/// assert_eq!(Ordering::Less, result);
///
/// let result = 1.cmp(&1);
/// assert_eq!(Ordering::Equal, result);
///
/// let result = 2.cmp(&1);
/// assert_eq!(Ordering::Greater, result);
/// ```
#[derive(Clone, Copy, PartialEq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Ordering {
    /// An ordering where a compared value is less [than another].
    #[stable(feature = "rust1", since = "1.0.0")]
    Less = -1,
    /// An ordering where a compared value is equal [to another].
    #[stable(feature = "rust1", since = "1.0.0")]
    Equal = 0,
    /// An ordering where a compared value is greater [than another].
    #[stable(feature = "rust1", since = "1.0.0")]
    Greater = 1,
}

impl Ordering {
    unsafe fn from_i8_unchecked(v: i8) -> Ordering {
        mem::transmute(v)
    }

    /// Reverse the `Ordering`.
    ///
    /// * `Less` becomes `Greater`.
    /// * `Greater` becomes `Less`.
    /// * `Equal` becomes `Equal`.
    ///
    /// # Examples
    ///
    /// Basic behavior:
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.reverse(), Ordering::Greater);
    /// assert_eq!(Ordering::Equal.reverse(), Ordering::Equal);
    /// assert_eq!(Ordering::Greater.reverse(), Ordering::Less);
    /// ```
    ///
    /// This method can be used to reverse a comparison:
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// let mut data: &mut [_] = &mut [2, 10, 5, 8];
    ///
    /// // sort the array from largest to smallest.
    /// data.sort_by(|a, b| a.cmp(b).reverse());
    ///
    /// let b: &mut [_] = &mut [10, 8, 5, 2];
    /// assert!(data == b);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reverse(self) -> Ordering {
        unsafe {
            // this compiles really nicely (to a single instruction);
            // an explicit match has a pile of branches and
            // comparisons.
            //
            // NB. it is safe because of the explicit discriminants
            // given above.
            Ordering::from_i8_unchecked(-(self as i8))
        }
    }
}

/// Trait for types that form a [total order](https://en.wikipedia.org/wiki/Total_order).
///
/// An order is a total order if it is (for all `a`, `b` and `c`):
///
/// - total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true; and
/// - transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
///
/// When this trait is `derive`d, it produces a lexicographic ordering.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Ord: Eq + PartialOrd<Self> {
    /// This method returns an `Ordering` between `self` and `other`.
    ///
    /// By convention, `self.cmp(&other)` returns the ordering matching the expression
    /// `self <operator> other` if true.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(5.cmp(&10), Ordering::Less);
    /// assert_eq!(10.cmp(&5), Ordering::Greater);
    /// assert_eq!(5.cmp(&5), Ordering::Equal);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn cmp(&self, other: &Self) -> Ordering;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for Ordering {}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for Ordering {
    #[inline]
    fn cmp(&self, other: &Ordering) -> Ordering {
        (*self as i32).cmp(&(*other as i32))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for Ordering {
    #[inline]
    fn partial_cmp(&self, other: &Ordering) -> Option<Ordering> {
        (*self as i32).partial_cmp(&(*other as i32))
    }
}

/// Trait for values that can be compared for a sort-order.
///
/// The comparison must satisfy, for all `a`, `b` and `c`:
///
/// - antisymmetry: if `a < b` then `!(a > b)`, as well as `a > b` implying `!(a < b)`; and
/// - transitivity: `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
///
/// Note that these requirements mean that the trait itself must be implemented symmetrically and
/// transitively: if `T: PartialOrd<U>` and `U: PartialOrd<V>` then `U: PartialOrd<T>` and `T:
/// PartialOrd<V>`.
///
/// PartialOrd only requires implementation of the `partial_cmp` method, with the others generated
/// from default implementations.
///
/// However it remains possible to implement the others separately for types which do not have a
/// total order. For example, for floating point numbers, `NaN < 0 == false` and `NaN >= 0 ==
/// false` (cf. IEEE 754-2008 section 5.11).
#[lang = "ord"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait PartialOrd<Rhs: ?Sized = Self>: PartialEq<Rhs> {
    /// This method returns an ordering between `self` and `other` values if one exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// let result = 1.0.partial_cmp(&2.0);
    /// assert_eq!(result, Some(Ordering::Less));
    ///
    /// let result = 1.0.partial_cmp(&1.0);
    /// assert_eq!(result, Some(Ordering::Equal));
    ///
    /// let result = 2.0.partial_cmp(&1.0);
    /// assert_eq!(result, Some(Ordering::Greater));
    /// ```
    ///
    /// When comparison is impossible:
    ///
    /// ```
    /// let result = std::f64::NAN.partial_cmp(&1.0);
    /// assert_eq!(result, None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;

    /// This method tests less than (for `self` and `other`) and is used by the `<` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// let result = 1.0 < 2.0;
    /// assert_eq!(result, true);
    ///
    /// let result = 2.0 < 1.0;
    /// assert_eq!(result, false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn lt(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Less) => true,
            _ => false,
        }
    }

    /// This method tests less than or equal to (for `self` and `other`) and is used by the `<=`
    /// operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = 1.0 <= 2.0;
    /// assert_eq!(result, true);
    ///
    /// let result = 2.0 <= 2.0;
    /// assert_eq!(result, true);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn le(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Less) | Some(Equal) => true,
            _ => false,
        }
    }

    /// This method tests greater than (for `self` and `other`) and is used by the `>` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = 1.0 > 2.0;
    /// assert_eq!(result, false);
    ///
    /// let result = 2.0 > 2.0;
    /// assert_eq!(result, false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn gt(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) => true,
            _ => false,
        }
    }

    /// This method tests greater than or equal to (for `self` and `other`) and is used by the `>=`
    /// operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = 2.0 >= 1.0;
    /// assert_eq!(result, true);
    ///
    /// let result = 2.0 >= 2.0;
    /// assert_eq!(result, true);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ge(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) | Some(Equal) => true,
            _ => false,
        }
    }
}

/// Compare and return the minimum of two values.
///
/// Returns the first argument if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// assert_eq!(1, cmp::min(1, 2));
/// assert_eq!(2, cmp::min(2, 2));
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn min<T: Ord>(v1: T, v2: T) -> T {
    if v1 <= v2 { v1 } else { v2 }
}

/// Compare and return the maximum of two values.
///
/// Returns the second argument if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// assert_eq!(2, cmp::max(1, 2));
/// assert_eq!(2, cmp::max(2, 2));
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn max<T: Ord>(v1: T, v2: T) -> T {
    if v2 >= v1 { v2 } else { v1 }
}

// Implementation of PartialEq, Eq, PartialOrd and Ord for primitive types
mod impls {
    use cmp::{PartialOrd, Ord, PartialEq, Eq, Ordering};
    use cmp::Ordering::{Less, Greater, Equal};
    use marker::Sized;
    use option::Option;
    use option::Option::{Some, None};

    macro_rules! partial_eq_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl PartialEq for $t {
                #[inline]
                fn eq(&self, other: &$t) -> bool { (*self) == (*other) }
                #[inline]
                fn ne(&self, other: &$t) -> bool { (*self) != (*other) }
            }
        )*)
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialEq for () {
        #[inline]
        fn eq(&self, _other: &()) -> bool { true }
        #[inline]
        fn ne(&self, _other: &()) -> bool { false }
    }

    partial_eq_impl! {
        bool char usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64
    }

    macro_rules! eq_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Eq for $t {}
        )*)
    }

    eq_impl! { () bool char usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

    macro_rules! partial_ord_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl PartialOrd for $t {
                #[inline]
                fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                    match (self <= other, self >= other) {
                        (false, false) => None,
                        (false, true) => Some(Greater),
                        (true, false) => Some(Less),
                        (true, true) => Some(Equal),
                    }
                }
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
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialOrd for () {
        #[inline]
        fn partial_cmp(&self, _: &()) -> Option<Ordering> {
            Some(Equal)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialOrd for bool {
        #[inline]
        fn partial_cmp(&self, other: &bool) -> Option<Ordering> {
            (*self as u8).partial_cmp(&(*other as u8))
        }
    }

    partial_ord_impl! { f32 f64 }

    macro_rules! ord_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl PartialOrd for $t {
                #[inline]
                fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
                #[inline]
                fn lt(&self, other: &$t) -> bool { (*self) < (*other) }
                #[inline]
                fn le(&self, other: &$t) -> bool { (*self) <= (*other) }
                #[inline]
                fn ge(&self, other: &$t) -> bool { (*self) >= (*other) }
                #[inline]
                fn gt(&self, other: &$t) -> bool { (*self) > (*other) }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl Ord for $t {
                #[inline]
                fn cmp(&self, other: &$t) -> Ordering {
                    if *self == *other { Equal }
                    else if *self < *other { Less }
                    else { Greater }
                }
            }
        )*)
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Ord for () {
        #[inline]
        fn cmp(&self, _other: &()) -> Ordering { Equal }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Ord for bool {
        #[inline]
        fn cmp(&self, other: &bool) -> Ordering {
            (*self as u8).cmp(&(*other as u8))
        }
    }

    ord_impl! { char usize u8 u16 u32 u64 isize i8 i16 i32 i64 }

    // & pointers

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<&'b B> for &'a A where A: PartialEq<B> {
        #[inline]
        fn eq(&self, other: & &'b B) -> bool { PartialEq::eq(*self, *other) }
        #[inline]
        fn ne(&self, other: & &'b B) -> bool { PartialEq::ne(*self, *other) }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, 'b, A: ?Sized, B: ?Sized> PartialOrd<&'b B> for &'a A where A: PartialOrd<B> {
        #[inline]
        fn partial_cmp(&self, other: &&'b B) -> Option<Ordering> {
            PartialOrd::partial_cmp(*self, *other)
        }
        #[inline]
        fn lt(&self, other: & &'b B) -> bool { PartialOrd::lt(*self, *other) }
        #[inline]
        fn le(&self, other: & &'b B) -> bool { PartialOrd::le(*self, *other) }
        #[inline]
        fn ge(&self, other: & &'b B) -> bool { PartialOrd::ge(*self, *other) }
        #[inline]
        fn gt(&self, other: & &'b B) -> bool { PartialOrd::gt(*self, *other) }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, A: ?Sized> Ord for &'a A where A: Ord {
        #[inline]
        fn cmp(&self, other: & &'a A) -> Ordering { Ord::cmp(*self, *other) }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, A: ?Sized> Eq for &'a A where A: Eq {}

    // &mut pointers

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<&'b mut B> for &'a mut A where A: PartialEq<B> {
        #[inline]
        fn eq(&self, other: &&'b mut B) -> bool { PartialEq::eq(*self, *other) }
        #[inline]
        fn ne(&self, other: &&'b mut B) -> bool { PartialEq::ne(*self, *other) }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, 'b, A: ?Sized, B: ?Sized> PartialOrd<&'b mut B> for &'a mut A where A: PartialOrd<B> {
        #[inline]
        fn partial_cmp(&self, other: &&'b mut B) -> Option<Ordering> {
            PartialOrd::partial_cmp(*self, *other)
        }
        #[inline]
        fn lt(&self, other: &&'b mut B) -> bool { PartialOrd::lt(*self, *other) }
        #[inline]
        fn le(&self, other: &&'b mut B) -> bool { PartialOrd::le(*self, *other) }
        #[inline]
        fn ge(&self, other: &&'b mut B) -> bool { PartialOrd::ge(*self, *other) }
        #[inline]
        fn gt(&self, other: &&'b mut B) -> bool { PartialOrd::gt(*self, *other) }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, A: ?Sized> Ord for &'a mut A where A: Ord {
        #[inline]
        fn cmp(&self, other: &&'a mut A) -> Ordering { Ord::cmp(*self, *other) }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, A: ?Sized> Eq for &'a mut A where A: Eq {}

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<&'b mut B> for &'a A where A: PartialEq<B> {
        #[inline]
        fn eq(&self, other: &&'b mut B) -> bool { PartialEq::eq(*self, *other) }
        #[inline]
        fn ne(&self, other: &&'b mut B) -> bool { PartialEq::ne(*self, *other) }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, 'b, A: ?Sized, B: ?Sized> PartialEq<&'b B> for &'a mut A where A: PartialEq<B> {
        #[inline]
        fn eq(&self, other: &&'b B) -> bool { PartialEq::eq(*self, *other) }
        #[inline]
        fn ne(&self, other: &&'b B) -> bool { PartialEq::ne(*self, *other) }
    }
}
