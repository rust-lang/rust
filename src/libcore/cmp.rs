// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functionality for ordering and relational operations.
//!
//! This module defines four traits related to ordering and comparisons:
//!
//! * [`PartialEq`]: which overloads the `==` and `!=` operators,
//! * [`PartialOrd`] (requires `PartialEq`): which overloads the `<`, `<=`, `>`, 
//!   and `>=` operators,
//!
//! state that values of a type can be compared for equality, or ordered, but that
//! not all values of the type can be compared to all other values (more on this below).
//!
//! ```
//! let x: u32 = 0;
//! let y: u32 = 1;
//!
//! // these two lines are equivalent
//! assert_eq!(x < y, true);
//! assert_eq!(x.lt(&y), true);
//!
//! // these two lines are also equivalent
//! assert_eq!(x == y, false);
//! assert_eq!(x.eq(&y), false);
//! ```
//!
//! This module also provides the following two traits:
//!
//! * [`Eq`]: which requires `PartialEq`, states that all values of a type can be
//!   compared to all others,
//! * [`Ord`]: which requires `PartialOrd` and `Eq`, states that the ordering
//!   relations expressed by the comparison operators form all total ordering relations 
//!   (more on this below).
//!
//! The distinction between these traits is important: it tell us what it _means_ for a 
//! sequence of values to be ordered.
//!
//! Two commonly encountered examples are integers and floating-point numbers. For example, every [`i32`] 
//! value is comparable to every other. If you can tell that two `i32` values are distinc, then these two
//! values never compare equal, and if you sorte a sequence of `i32` values like `[1, 1, 0, -5]` using `<=`,
//! doing so will always produce the same unique result: `[-5, 0, 1, 1]`. We say that `<=` imposes a _total_
//! order, and denote this by implementing `Ord` for `i32`.
//!
//! However, none of this holds for IEEE-754 floating-point numbers like `f32`. There is a special `f32` value 
//! called "Not a Number" (`NaN`) for which any relational operation alwys return false, that is: 
//! `a == NaN` is always `false`, even when `a` is `NaN`. For this reason `f32` only implements `PartialOrd`, and it 
//! does not implement neither `Eq` nor `Ord`. What does this mean in practice? It means two things:
//!
//! * there are sequences of `f32`s that we can order using `<=` and will always produce the same unique 
//!   result: sorting `[3.0, 5.0, 1.0]` into `[1.0, 3.0, 5.0]`.
//!
//! * there are also sequences of `f32`s that when ordered using `<=` will produce a different result depending
//!  on which algorithm we use to sort them: we can sort `[-0.0, NaN, +0.0] into all its permutations 
//! (`[-0.0, +0.0, NaN]`, `[+0.0, -0.0, NaN]`, `[NaN, +0.0, -0.0]`, etc. are all "sorted" under `<=`).
//!
//! Another example is the "subset relation" on sets in which, for sets `a` and `b`, `a < b` is true if and only 
//! if `a` is a proper subset of `b` (e.g. `{4, 9} < {1, 4, 5, 9}` is true). The problem is that when neither 
//! `a` is a subset of `b` nor `b` is a subset of `a` nor `a = b` (e.g. `{1, 3}` and `{1, 4}`), we cannot sensible 
//! compare `a` and `b`, so that `a < b`, `a == b`, and `a > b` all return false.
//!
//! The distinction between a _total_ order and a _partial_ order is important, and impacts what it means for
//! the values of a type to be "sorted". This impacts how it makes sense to use these values. For example,
//! if we sort a vector of `f32`s containing `NaN`s, does it make sense to use binary search to try to find a value
//! in that vector? Will it always work? Does it make sense to put `f32`s containing `NaN`s into a set? a hash table?
//! etc. Algorithms and data-structures that require a total order or total equality to work correctly can constrain
//! the types they accept with the [`Eq`] and [`Ord`] traits, while those for which a partial order or partial equality
//! is enough can use the `PartialEq` and `PartialOrd` traits.
//!
//! This leads us to the final question: which traits should a type implement ? There is no easy answer: a type should only 
//! implement the traits that make sense for the type, but this answer does not help. The different traits have
//! different semantic requirements in their operations, and if a type cannot satisfy these requirements then it probably
//! should not implement the trait. However, there are types that _could_ satisfy the requirements but for which it 
//! still might not make sense to implement the trait. For example, a `Point(i32,i32)` could implement `Ord` by using
//! a lexicographical-order, but whether this might make sense would depend on the application. Another example is 
//! floating-point numbers: the IEEE754 standard provides _multiple_ different orderings for floating-point numbers. The one
//! we have seen above is cheap to compute, but it is only a partial order. The IEEE754 standard also provides a 
//! total order for floating point numbers that is more expensive to compute. So which ordering should they implement? Arguably,
//! they should offer both orders and let the user choose, but they can only implement the trait once. For these kind of types,
/// for which multiple orderings exist, it might make sense to not implement any order at all, but use another mechanism 
//! (e.g. a wrapper type) to allow users to select the order they want. 
//!
//! [`PartialOrd`]: trait.PartialOrd.html
//! [`PartialEq`]: trait.PartialEq.html
//! [`Ord`]: trait.Ord.html
//! [`Eq`]: trait.Eq.html

#![stable(feature = "rust1", since = "1.0.0")]

use self::Ordering::*;

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
/// ## Derivable
///
/// This trait can be used with `#[derive]`. When `derive`d on structs, two
/// instances are equal if all fields are equal, and not equal if any fields
/// are not equal. When `derive`d on enums, each variant is equal to itself
/// and not equal to the other variants.
///
/// ## How can I implement `PartialEq`?
///
/// PartialEq only requires the `eq` method to be implemented; `ne` is defined
/// in terms of it by default. Any manual implementation of `ne` *must* respect
/// the rule that `eq` is a strict inverse of `ne`; that is, `!(a == b)` if and
/// only if `a != b`.
///
/// Implementations of `PartialEq`, `PartialOrd`, and `Ord` *must* agree with
/// each other. It's easy to accidentally make them disagree by deriving some
/// of the traits and manually implementing others.
///
/// An example implementation for a domain in which two books are considered
/// the same book if their ISBN matches, even if the formats differ:
///
/// ```
/// enum BookFormat { Paperback, Hardback, Ebook }
/// struct Book {
///     isbn: i32,
///     format: BookFormat,
/// }
///
/// impl PartialEq for Book {
///     fn eq(&self, other: &Book) -> bool {
///         self.isbn == other.isbn
///     }
/// }
///
/// let b1 = Book { isbn: 3, format: BookFormat::Paperback };
/// let b2 = Book { isbn: 3, format: BookFormat::Ebook };
/// let b3 = Book { isbn: 10, format: BookFormat::Paperback };
///
/// assert!(b1 == b2);
/// assert!(b1 != b3);
/// ```
///
/// # Examples
///
/// ```
/// let x: u32 = 0;
/// let y: u32 = 1;
///
/// assert_eq!(x == y, false);
/// assert_eq!(x.eq(&y), false);
/// ```
#[lang = "eq"]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = "==")]
#[doc(alias = "!=")]
#[rustc_on_unimplemented(
    message="can't compare `{Self}` with `{Rhs}`",
    label="no implementation for `{Self} == {Rhs}`",
)]
pub trait PartialEq<Rhs: ?Sized = Self> {
    /// This method tests for `self` and `other` values to be equal, and is used
    /// by `==`.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn eq(&self, other: &Rhs) -> bool;

    /// This method tests for `!=`.
    #[inline]
    #[must_use]
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
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`. When `derive`d, because `Eq` has
/// no extra methods, it is only informing the compiler that this is an
/// equivalence relation rather than a partial equivalence relation. Note that
/// the `derive` strategy requires all fields are `Eq`, which isn't
/// always desired.
///
/// ## How can I implement `Eq`?
///
/// If you cannot use the `derive` strategy, specify that your type implements
/// `Eq`, which has no methods:
///
/// ```
/// enum BookFormat { Paperback, Hardback, Ebook }
/// struct Book {
///     isbn: i32,
///     format: BookFormat,
/// }
/// impl PartialEq for Book {
///     fn eq(&self, other: &Book) -> bool {
///         self.isbn == other.isbn
///     }
/// }
/// impl Eq for Book {}
/// ```
#[doc(alias = "==")]
#[doc(alias = "!=")]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Eq: PartialEq<Self> {
    // this method is used solely by #[deriving] to assert
    // that every component of a type implements #[deriving]
    // itself, the current deriving infrastructure means doing this
    // assertion without using a method on this trait is nearly
    // impossible.
    //
    // This should never be implemented by hand.
    #[doc(hidden)]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn assert_receiver_is_total_eq(&self) {}
}

// FIXME: this struct is used solely by #[derive] to
// assert that every component of a type implements Eq.
//
// This struct should never appear in user code.
#[doc(hidden)]
#[allow(missing_debug_implementations)]
#[unstable(feature = "derive_eq",
           reason = "deriving hack, should not be public",
           issue = "0")]
pub struct AssertParamIsEq<T: Eq + ?Sized> { _field: ::marker::PhantomData<T> }

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
#[derive(Clone, Copy, PartialEq, Debug, Hash)]
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
    /// Reverses the `Ordering`.
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
        match self {
            Less => Greater,
            Equal => Equal,
            Greater => Less,
        }
    }

    /// Chains two orderings.
    ///
    /// Returns `self` when it's not `Equal`. Otherwise returns `other`.
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// let result = Ordering::Equal.then(Ordering::Less);
    /// assert_eq!(result, Ordering::Less);
    ///
    /// let result = Ordering::Less.then(Ordering::Equal);
    /// assert_eq!(result, Ordering::Less);
    ///
    /// let result = Ordering::Less.then(Ordering::Greater);
    /// assert_eq!(result, Ordering::Less);
    ///
    /// let result = Ordering::Equal.then(Ordering::Equal);
    /// assert_eq!(result, Ordering::Equal);
    ///
    /// let x: (i64, i64, i64) = (1, 2, 7);
    /// let y: (i64, i64, i64) = (1, 5, 3);
    /// let result = x.0.cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2));
    ///
    /// assert_eq!(result, Ordering::Less);
    /// ```
    #[inline]
    #[stable(feature = "ordering_chaining", since = "1.17.0")]
    pub fn then(self, other: Ordering) -> Ordering {
        match self {
            Equal => other,
            _ => self,
        }
    }

    /// Chains the ordering with the given function.
    ///
    /// Returns `self` when it's not `Equal`. Otherwise calls `f` and returns
    /// the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// let result = Ordering::Equal.then_with(|| Ordering::Less);
    /// assert_eq!(result, Ordering::Less);
    ///
    /// let result = Ordering::Less.then_with(|| Ordering::Equal);
    /// assert_eq!(result, Ordering::Less);
    ///
    /// let result = Ordering::Less.then_with(|| Ordering::Greater);
    /// assert_eq!(result, Ordering::Less);
    ///
    /// let result = Ordering::Equal.then_with(|| Ordering::Equal);
    /// assert_eq!(result, Ordering::Equal);
    ///
    /// let x: (i64, i64, i64) = (1, 2, 7);
    /// let y: (i64, i64, i64)  = (1, 5, 3);
    /// let result = x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)).then_with(|| x.2.cmp(&y.2));
    ///
    /// assert_eq!(result, Ordering::Less);
    /// ```
    #[inline]
    #[stable(feature = "ordering_chaining", since = "1.17.0")]
    pub fn then_with<F: FnOnce() -> Ordering>(self, f: F) -> Ordering {
        match self {
            Equal => f(),
            _ => self,
        }
    }
}

/// A helper struct for reverse ordering.
///
/// This struct is a helper to be used with functions like `Vec::sort_by_key` and
/// can be used to reverse order a part of a key.
///
/// Example usage:
///
/// ```
/// use std::cmp::Reverse;
///
/// let mut v = vec![1, 2, 3, 4, 5, 6];
/// v.sort_by_key(|&num| (num > 3, Reverse(num)));
/// assert_eq!(v, vec![3, 2, 1, 6, 5, 4]);
/// ```
#[derive(PartialEq, Eq, Debug, Copy, Clone, Default, Hash)]
#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
pub struct Reverse<T>(#[stable(feature = "reverse_cmp_key", since = "1.19.0")] pub T);

#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
impl<T: PartialOrd> PartialOrd for Reverse<T> {
    #[inline]
    fn partial_cmp(&self, other: &Reverse<T>) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool { other.0 < self.0 }
    #[inline]
    fn le(&self, other: &Self) -> bool { other.0 <= self.0 }
    #[inline]
    fn ge(&self, other: &Self) -> bool { other.0 >= self.0 }
    #[inline]
    fn gt(&self, other: &Self) -> bool { other.0 > self.0 }
}

#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
impl<T: Ord> Ord for Reverse<T> {
    #[inline]
    fn cmp(&self, other: &Reverse<T>) -> Ordering {
        other.0.cmp(&self.0)
    }
}

/// Trait for types that form a [total order](https://en.wikipedia.org/wiki/Total_order).
///
/// An order is a total order if it is (for all `a`, `b` and `c`):
///
/// - total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true; and
/// - transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`. When `derive`d on structs, it will produce a
/// lexicographic ordering based on the top-to-bottom declaration order of the struct's members.
/// When `derive`d on enums, variants are ordered by their top-to-bottom declaration order.
///
/// ## How can I implement `Ord`?
///
/// `Ord` requires that the type also be `PartialOrd` and `Eq` (which requires `PartialEq`).
///
/// Then you must define an implementation for `cmp()`. You may find it useful to use
/// `cmp()` on your type's fields.
///
/// Implementations of `PartialEq`, `PartialOrd`, and `Ord` *must* agree with each other. It's
/// easy to accidentally make them disagree by deriving some of the traits and manually
/// implementing others.
///
/// Here's an example where you want to sort people by height only, disregarding `id`
/// and `name`:
///
/// ```
/// use std::cmp::Ordering;
///
/// #[derive(Eq)]
/// struct Person {
///     id: u32,
///     name: String,
///     height: u32,
/// }
///
/// impl Ord for Person {
///     fn cmp(&self, other: &Person) -> Ordering {
///         self.height.cmp(&other.height)
///     }
/// }
///
/// impl PartialOrd for Person {
///     fn partial_cmp(&self, other: &Person) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// impl PartialEq for Person {
///     fn eq(&self, other: &Person) -> bool {
///         self.height == other.height
///     }
/// }
/// ```
#[lang = "ord"]
#[doc(alias = "<")]
#[doc(alias = ">")]
#[doc(alias = "<=")]
#[doc(alias = ">=")]
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

    /// Compares and returns the maximum of two values.
    ///
    /// Returns the second argument if the comparison determines them to be equal.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(2, 1.max(2));
    /// assert_eq!(2, 2.max(2));
    /// ```
    #[stable(feature = "ord_max_min", since = "1.21.0")]
    #[inline]
    fn max(self, other: Self) -> Self
    where Self: Sized {
        if other >= self { other } else { self }
    }

    /// Compares and returns the minimum of two values.
    ///
    /// Returns the first argument if the comparison determines them to be equal.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1, 1.min(2));
    /// assert_eq!(2, 2.min(2));
    /// ```
    #[stable(feature = "ord_max_min", since = "1.21.0")]
    #[inline]
    fn min(self, other: Self) -> Self
    where Self: Sized {
        if self <= other { self } else { other }
    }
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

/// Trait for values that can be partially ordered.
///
/// This traits defines the operators `<`, `>`, `<=`, and `>=` which compare two values and return `bool`, 
/// where `<` and `>` form [_strict_ partial ordering relations][wiki_porder].
///
/// The `PartialOrd` trait  has fewer requirements of the ordering relation than Ord. In particular, `partial_cmp` 
/// can return `None` if two elements cannot be compared. That means that `PartialOrd` can be implemented 
/// for types with incomparable values like floating-point numbers (`NaN`) and for ordering relations like 
/// subset relations.
///
/// It extends the partial equivalence relation provided by `PartialEq` (`==`) with `partial_cmp(a, b) -> Option<Ordering>`, 
/// which is a [trichotomy](https://en.wikipedia.org/wiki/Trichotomy_(mathematics)) of the ordering relation when 
/// its result is `Some`:
///
/// * `a < b` if and only if `partial_cmp(a, b) == Some(Less)`
/// * `a > b` if and only if `partial_cmp(a, b) == Some(Greater)`
/// * `a == b` if and only if `partial_cmp(a, b) == Some(Equal)`
///
/// and the absence of an ordering between `a` and `b` if and only if `partial_cmp(a, b) == None`. 
/// Furthermore, this trait defines `<=` as `a < b || a == b` and `>=` as `a > b || a == b`.
///
/// The comparisons must satisfy, for all `a`, `b`, and `c`:
///
/// * _transitivity_: if `a < b` and `b < c` then `a < c`
/// * _duality_: if `a < b` then `b > a`
///
/// The `lt` (`<`), `le` (`<=`), `gt` (`>`), and `ge` (`>=`) methods are implemented in terms of `partial_cmp` 
/// according to these rules. The default implementations can be overridden for performance reasons, but manual 
/// implementations must satisfy the rules above.
///
/// From these rules it follows that `PartialOrd` must be implemented symmetrically and transitively.
/// That is, for two distinct types `T` and `U`, `T: PartialOrd<U>` requires the following trait 
/// implementations to exist and satisfy these rules: `T: PartialOrd<T>`, `U: PartialOrd<T>`, 
/// and `U: PartialOrd<U>`. If for a third distinct type `V` `T: PartialOrd<V>` is provided, then
/// the following implementations should be implemented accordingly: `V: PartialOrd<V>`, 
/// `V: PartialOrd<T>`, `U: PartialOrd<V>`, and `V: PartialOrd<U>`.
/// 
/// The following corollaries follow from transitivity of `<`, duality, and from the definition of `<` et al. 
/// in terms of the same `partial_cmp`:
///
/// * irreflexivity of `<`: `!(a < a)`
/// * transitivity of `>`: if `a > b` and `b > c` then `a > c`
/// * asymmetry of `<`: if `a < b` then `!(b < a)`
///
/// Stronger ordering relations can be expressed by using the `Eq` and `Ord` traits, where the `PartialOrd` 
/// methods provide:
///
/// * a [_non-strict_ partial ordering relation](https://en.wikipedia.org/wiki/Partially_ordered_set#Strict_and_non-strict_partial_orders)
///   if `T: PartialOrd + Eq`
/// * a [total ordering relation](https://en.wikipedia.org/wiki/Total_order) if `T: Ord`
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`. When `derive`d on structs, it will produce a
/// lexicographic ordering based on the top-to-bottom declaration order of the struct's members.
/// When `derive`d on enums, variants are ordered by their top-to-bottom declaration order.
///
/// ## How can I implement `PartialOrd`?
///
/// `PartialOrd` only requires implementation of the `partial_cmp` method, with the others
/// generated from default implementations.
///
/// However it remains possible to implement the others separately for types which do not have a
/// total order. For example, for floating point numbers, `NaN < 0 == false` and `NaN >= 0 ==
/// false` (cf. IEEE 754-2008 section 5.11).
///
/// `PartialOrd` requires your type to be `PartialEq`.
///
/// Implementations of `PartialEq`, `PartialOrd`, and `Ord` *must* agree with each other. It's
/// easy to accidentally make them disagree by deriving some of the traits and manually
/// implementing others.
///
/// If your type is `Ord`, you can implement `partial_cmp()` by using `cmp()`:
///
/// ```
/// use std::cmp::Ordering;
///
/// #[derive(Eq)]
/// struct Person {
///     id: u32,
///     name: String,
///     height: u32,
/// }
///
/// impl PartialOrd for Person {
///     fn partial_cmp(&self, other: &Person) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// impl Ord for Person {
///     fn cmp(&self, other: &Person) -> Ordering {
///         self.height.cmp(&other.height)
///     }
/// }
///
/// impl PartialEq for Person {
///     fn eq(&self, other: &Person) -> bool {
///         self.height == other.height
///     }
/// }
/// ```
///
/// You may also find it useful to use `partial_cmp()` on your type's fields. Here
/// is an example of `Person` types who have a floating-point `height` field that
/// is the only field to be used for sorting:
///
/// ```
/// use std::cmp::Ordering;
///
/// struct Person {
///     id: u32,
///     name: String,
///     height: f64,
/// }
///
/// impl PartialOrd for Person {
///     fn partial_cmp(&self, other: &Person) -> Option<Ordering> {
///         self.height.partial_cmp(&other.height)
///     }
/// }
///
/// impl PartialEq for Person {
///     fn eq(&self, other: &Person) -> bool {
///         self.height == other.height
///     }
/// }
/// ```
///
/// # Examples
///
/// ```
/// let x : u32 = 0;
/// let y : u32 = 1;
///
/// assert_eq!(x < y, true);
/// assert_eq!(x.lt(&y), true);
/// ```
///
/// [wiki_porder]: https://en.wikipedia.org/wiki/Partially_ordered_set#Strict_and_non-strict_partial_orders
#[lang = "partial_ord"]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = ">")]
#[doc(alias = "<")]
#[doc(alias = "<=")]
#[doc(alias = ">=")]
#[rustc_on_unimplemented(
    message="can't compare `{Self}` with `{Rhs}`",
    label="no implementation for `{Self} < {Rhs}` and `{Self} > {Rhs}`",
)]
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;

    /// This method tests less than (for `self` and `other`) and is used by the `<` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let result = 1.0 < 2.0;
    /// assert_eq!(result, true);
    ///
    /// let result = 2.0 < 1.0;
    /// assert_eq!(result, false);
    /// ```
    #[inline]
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn ge(&self, other: &Rhs) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) | Some(Equal) => true,
            _ => false,
        }
    }
}

/// Compares and returns the minimum of two values.
///
/// Returns the first argument if the comparison determines them to be equal.
///
/// Internally uses an alias to `Ord::min`.
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
    v1.min(v2)
}

/// Compares and returns the maximum of two values.
///
/// Returns the second argument if the comparison determines them to be equal.
///
/// Internally uses an alias to `Ord::max`.
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
    v1.max(v2)
}

// Implementation of PartialEq, Eq, PartialOrd and Ord for primitive types
mod impls {
    use cmp::Ordering::{self, Less, Greater, Equal};

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
        bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64
    }

    macro_rules! eq_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Eq for $t {}
        )*)
    }

    eq_impl! { () bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

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

    ord_impl! { char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

    #[unstable(feature = "never_type", issue = "35121")]
    impl PartialEq for ! {
        fn eq(&self, _: &!) -> bool {
            *self
        }
    }

    #[unstable(feature = "never_type", issue = "35121")]
    impl Eq for ! {}

    #[unstable(feature = "never_type", issue = "35121")]
    impl PartialOrd for ! {
        fn partial_cmp(&self, _: &!) -> Option<Ordering> {
            *self
        }
    }

    #[unstable(feature = "never_type", issue = "35121")]
    impl Ord for ! {
        fn cmp(&self, _: &!) -> Ordering {
            *self
        }
    }

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
