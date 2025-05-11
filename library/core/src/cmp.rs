//! Utilities for comparing and ordering values.
//!
//! This module contains various tools for comparing and ordering values. In
//! summary:
//!
//! * [`PartialEq<Rhs>`] overloads the `==` and `!=` operators. In cases where
//!   `Rhs` (the right hand side's type) is `Self`, this trait corresponds to a
//!   partial equivalence relation.
//! * [`Eq`] indicates that the overloaded `==` operator corresponds to an
//!   equivalence relation.
//! * [`Ord`] and [`PartialOrd`] are traits that allow you to define total and
//!   partial orderings between values, respectively. Implementing them overloads
//!   the `<`, `<=`, `>`, and `>=` operators.
//! * [`Ordering`] is an enum returned by the main functions of [`Ord`] and
//!   [`PartialOrd`], and describes an ordering of two values (less, equal, or
//!   greater).
//! * [`Reverse`] is a struct that allows you to easily reverse an ordering.
//! * [`max`] and [`min`] are functions that build off of [`Ord`] and allow you
//!   to find the maximum or minimum of two values.
//!
//! For more details, see the respective documentation of each item in the list.
//!
//! [`max`]: Ord::max
//! [`min`]: Ord::min

#![stable(feature = "rust1", since = "1.0.0")]

mod bytewise;
pub(crate) use bytewise::BytewiseEq;

use self::Ordering::*;
use crate::ops::ControlFlow;

/// Trait for comparisons using the equality operator.
///
/// Implementing this trait for types provides the `==` and `!=` operators for
/// those types.
///
/// `x.eq(y)` can also be written `x == y`, and `x.ne(y)` can be written `x != y`.
/// We use the easier-to-read infix notation in the remainder of this documentation.
///
/// This trait allows for comparisons using the equality operator, for types
/// that do not have a full equivalence relation. For example, in floating point
/// numbers `NaN != NaN`, so floating point types implement `PartialEq` but not
/// [`trait@Eq`]. Formally speaking, when `Rhs == Self`, this trait corresponds
/// to a [partial equivalence relation].
///
/// [partial equivalence relation]: https://en.wikipedia.org/wiki/Partial_equivalence_relation
///
/// Implementations must ensure that `eq` and `ne` are consistent with each other:
///
/// - `a != b` if and only if `!(a == b)`.
///
/// The default implementation of `ne` provides this consistency and is almost
/// always sufficient. It should not be overridden without very good reason.
///
/// If [`PartialOrd`] or [`Ord`] are also implemented for `Self` and `Rhs`, their methods must also
/// be consistent with `PartialEq` (see the documentation of those traits for the exact
/// requirements). It's easy to accidentally make them disagree by deriving some of the traits and
/// manually implementing others.
///
/// The equality relation `==` must satisfy the following conditions
/// (for all `a`, `b`, `c` of type `A`, `B`, `C`):
///
/// - **Symmetry**: if `A: PartialEq<B>` and `B: PartialEq<A>`, then **`a == b`
///   implies `b == a`**; and
///
/// - **Transitivity**: if `A: PartialEq<B>` and `B: PartialEq<C>` and `A:
///   PartialEq<C>`, then **`a == b` and `b == c` implies `a == c`**.
///   This must also work for longer chains, such as when `A: PartialEq<B>`, `B: PartialEq<C>`,
///   `C: PartialEq<D>`, and `A: PartialEq<D>` all exist.
///
/// Note that the `B: PartialEq<A>` (symmetric) and `A: PartialEq<C>`
/// (transitive) impls are not forced to exist, but these requirements apply
/// whenever they do exist.
///
/// Violating these requirements is a logic error. The behavior resulting from a logic error is not
/// specified, but users of the trait must ensure that such logic errors do *not* result in
/// undefined behavior. This means that `unsafe` code **must not** rely on the correctness of these
/// methods.
///
/// ## Cross-crate considerations
///
/// Upholding the requirements stated above can become tricky when one crate implements `PartialEq`
/// for a type of another crate (i.e., to allow comparing one of its own types with a type from the
/// standard library). The recommendation is to never implement this trait for a foreign type. In
/// other words, such a crate should do `impl PartialEq<ForeignType> for LocalType`, but it should
/// *not* do `impl PartialEq<LocalType> for ForeignType`.
///
/// This avoids the problem of transitive chains that criss-cross crate boundaries: for all local
/// types `T`, you may assume that no other crate will add `impl`s that allow comparing `T == U`. In
/// other words, if other crates add `impl`s that allow building longer transitive chains `U1 == ...
/// == T == V1 == ...`, then all the types that appear to the right of `T` must be types that the
/// crate defining `T` already knows about. This rules out transitive chains where downstream crates
/// can add new `impl`s that "stitch together" comparisons of foreign types in ways that violate
/// transitivity.
///
/// Not having such foreign `impl`s also avoids forward compatibility issues where one crate adding
/// more `PartialEq` implementations can cause build failures in downstream crates.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`. When `derive`d on structs, two
/// instances are equal if all fields are equal, and not equal if any fields
/// are not equal. When `derive`d on enums, two instances are equal if they
/// are the same variant and all fields are equal.
///
/// ## How can I implement `PartialEq`?
///
/// An example implementation for a domain in which two books are considered
/// the same book if their ISBN matches, even if the formats differ:
///
/// ```
/// enum BookFormat {
///     Paperback,
///     Hardback,
///     Ebook,
/// }
///
/// struct Book {
///     isbn: i32,
///     format: BookFormat,
/// }
///
/// impl PartialEq for Book {
///     fn eq(&self, other: &Self) -> bool {
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
/// ## How can I compare two different types?
///
/// The type you can compare with is controlled by `PartialEq`'s type parameter.
/// For example, let's tweak our previous code a bit:
///
/// ```
/// // The derive implements <BookFormat> == <BookFormat> comparisons
/// #[derive(PartialEq)]
/// enum BookFormat {
///     Paperback,
///     Hardback,
///     Ebook,
/// }
///
/// struct Book {
///     isbn: i32,
///     format: BookFormat,
/// }
///
/// // Implement <Book> == <BookFormat> comparisons
/// impl PartialEq<BookFormat> for Book {
///     fn eq(&self, other: &BookFormat) -> bool {
///         self.format == *other
///     }
/// }
///
/// // Implement <BookFormat> == <Book> comparisons
/// impl PartialEq<Book> for BookFormat {
///     fn eq(&self, other: &Book) -> bool {
///         *self == other.format
///     }
/// }
///
/// let b1 = Book { isbn: 3, format: BookFormat::Paperback };
///
/// assert!(b1 == BookFormat::Paperback);
/// assert!(BookFormat::Ebook != b1);
/// ```
///
/// By changing `impl PartialEq for Book` to `impl PartialEq<BookFormat> for Book`,
/// we allow `BookFormat`s to be compared with `Book`s.
///
/// A comparison like the one above, which ignores some fields of the struct,
/// can be dangerous. It can easily lead to an unintended violation of the
/// requirements for a partial equivalence relation. For example, if we kept
/// the above implementation of `PartialEq<Book>` for `BookFormat` and added an
/// implementation of `PartialEq<Book>` for `Book` (either via a `#[derive]` or
/// via the manual implementation from the first example) then the result would
/// violate transitivity:
///
/// ```should_panic
/// #[derive(PartialEq)]
/// enum BookFormat {
///     Paperback,
///     Hardback,
///     Ebook,
/// }
///
/// #[derive(PartialEq)]
/// struct Book {
///     isbn: i32,
///     format: BookFormat,
/// }
///
/// impl PartialEq<BookFormat> for Book {
///     fn eq(&self, other: &BookFormat) -> bool {
///         self.format == *other
///     }
/// }
///
/// impl PartialEq<Book> for BookFormat {
///     fn eq(&self, other: &Book) -> bool {
///         *self == other.format
///     }
/// }
///
/// fn main() {
///     let b1 = Book { isbn: 1, format: BookFormat::Paperback };
///     let b2 = Book { isbn: 2, format: BookFormat::Paperback };
///
///     assert!(b1 == BookFormat::Paperback);
///     assert!(BookFormat::Paperback == b2);
///
///     // The following should hold by transitivity but doesn't.
///     assert!(b1 == b2); // <-- PANICS
/// }
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
///
/// [`eq`]: PartialEq::eq
/// [`ne`]: PartialEq::ne
#[lang = "eq"]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = "==")]
#[doc(alias = "!=")]
#[rustc_on_unimplemented(
    message = "can't compare `{Self}` with `{Rhs}`",
    label = "no implementation for `{Self} == {Rhs}`",
    append_const_msg
)]
#[rustc_diagnostic_item = "PartialEq"]
pub trait PartialEq<Rhs: ?Sized = Self> {
    /// Tests for `self` and `other` values to be equal, and is used by `==`.
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialeq_eq"]
    fn eq(&self, other: &Rhs) -> bool;

    /// Tests for `!=`. The default implementation is almost always sufficient,
    /// and should not be overridden without very good reason.
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialeq_ne"]
    fn ne(&self, other: &Rhs) -> bool {
        !self.eq(other)
    }
}

/// Derive macro generating an impl of the trait [`PartialEq`].
/// The behavior of this macro is described in detail [here](PartialEq#derivable).
#[rustc_builtin_macro]
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow_internal_unstable(core_intrinsics, structural_match)]
pub macro PartialEq($item:item) {
    /* compiler built-in */
}

/// Trait for comparisons corresponding to [equivalence relations](
/// https://en.wikipedia.org/wiki/Equivalence_relation).
///
/// The primary difference to [`PartialEq`] is the additional requirement for reflexivity. A type
/// that implements [`PartialEq`] guarantees that for all `a`, `b` and `c`:
///
/// - symmetric: `a == b` implies `b == a` and `a != b` implies `!(a == b)`
/// - transitive: `a == b` and `b == c` implies `a == c`
///
/// `Eq`, which builds on top of [`PartialEq`] also implies:
///
/// - reflexive: `a == a`
///
/// This property cannot be checked by the compiler, and therefore `Eq` is a trait without methods.
///
/// Violating this property is a logic error. The behavior resulting from a logic error is not
/// specified, but users of the trait must ensure that such logic errors do *not* result in
/// undefined behavior. This means that `unsafe` code **must not** rely on the correctness of these
/// methods.
///
/// Floating point types such as [`f32`] and [`f64`] implement only [`PartialEq`] but *not* `Eq`
/// because `NaN` != `NaN`.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`. When `derive`d, because `Eq` has no extra methods, it
/// is only informing the compiler that this is an equivalence relation rather than a partial
/// equivalence relation. Note that the `derive` strategy requires all fields are `Eq`, which isn't
/// always desired.
///
/// ## How can I implement `Eq`?
///
/// If you cannot use the `derive` strategy, specify that your type implements `Eq`, which has no
/// extra methods:
///
/// ```
/// enum BookFormat {
///     Paperback,
///     Hardback,
///     Ebook,
/// }
///
/// struct Book {
///     isbn: i32,
///     format: BookFormat,
/// }
///
/// impl PartialEq for Book {
///     fn eq(&self, other: &Self) -> bool {
///         self.isbn == other.isbn
///     }
/// }
///
/// impl Eq for Book {}
/// ```
#[doc(alias = "==")]
#[doc(alias = "!=")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Eq"]
pub trait Eq: PartialEq<Self> {
    // this method is used solely by `impl Eq or #[derive(Eq)]` to assert that every component of a
    // type implements `Eq` itself. The current deriving infrastructure means doing this assertion
    // without using a method on this trait is nearly impossible.
    //
    // This should never be implemented by hand.
    #[doc(hidden)]
    #[coverage(off)]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn assert_receiver_is_total_eq(&self) {}
}

/// Derive macro generating an impl of the trait [`Eq`].
#[rustc_builtin_macro]
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow_internal_unstable(core_intrinsics, derive_eq, structural_match)]
#[allow_internal_unstable(coverage_attribute)]
pub macro Eq($item:item) {
    /* compiler built-in */
}

// FIXME: this struct is used solely by #[derive] to
// assert that every component of a type implements Eq.
//
// This struct should never appear in user code.
#[doc(hidden)]
#[allow(missing_debug_implementations)]
#[unstable(feature = "derive_eq", reason = "deriving hack, should not be public", issue = "none")]
pub struct AssertParamIsEq<T: Eq + ?Sized> {
    _field: crate::marker::PhantomData<T>,
}

/// An `Ordering` is the result of a comparison between two values.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
///
/// assert_eq!(1.cmp(&2), Ordering::Less);
///
/// assert_eq!(1.cmp(&1), Ordering::Equal);
///
/// assert_eq!(2.cmp(&1), Ordering::Greater);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
// This is a lang item only so that `BinOp::Cmp` in MIR can return it.
// It has no special behavior, but does require that the three variants
// `Less`/`Equal`/`Greater` remain `-1_i8`/`0_i8`/`+1_i8` respectively.
#[lang = "Ordering"]
#[repr(i8)]
pub enum Ordering {
    /// An ordering where a compared value is less than another.
    #[stable(feature = "rust1", since = "1.0.0")]
    Less = -1,
    /// An ordering where a compared value is equal to another.
    #[stable(feature = "rust1", since = "1.0.0")]
    Equal = 0,
    /// An ordering where a compared value is greater than another.
    #[stable(feature = "rust1", since = "1.0.0")]
    Greater = 1,
}

impl Ordering {
    #[inline]
    const fn as_raw(self) -> i8 {
        // FIXME(const-hack): just use `PartialOrd` against `Equal` once that's const
        crate::intrinsics::discriminant_value(&self)
    }

    /// Returns `true` if the ordering is the `Equal` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.is_eq(), false);
    /// assert_eq!(Ordering::Equal.is_eq(), true);
    /// assert_eq!(Ordering::Greater.is_eq(), false);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "ordering_helpers", since = "1.53.0")]
    #[stable(feature = "ordering_helpers", since = "1.53.0")]
    pub const fn is_eq(self) -> bool {
        // All the `is_*` methods are implemented as comparisons against zero
        // to follow how clang's libcxx implements their equivalents in
        // <https://github.com/llvm/llvm-project/blob/60486292b79885b7800b082754153202bef5b1f0/libcxx/include/__compare/is_eq.h#L23-L28>

        self.as_raw() == 0
    }

    /// Returns `true` if the ordering is not the `Equal` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.is_ne(), true);
    /// assert_eq!(Ordering::Equal.is_ne(), false);
    /// assert_eq!(Ordering::Greater.is_ne(), true);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "ordering_helpers", since = "1.53.0")]
    #[stable(feature = "ordering_helpers", since = "1.53.0")]
    pub const fn is_ne(self) -> bool {
        self.as_raw() != 0
    }

    /// Returns `true` if the ordering is the `Less` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.is_lt(), true);
    /// assert_eq!(Ordering::Equal.is_lt(), false);
    /// assert_eq!(Ordering::Greater.is_lt(), false);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "ordering_helpers", since = "1.53.0")]
    #[stable(feature = "ordering_helpers", since = "1.53.0")]
    pub const fn is_lt(self) -> bool {
        self.as_raw() < 0
    }

    /// Returns `true` if the ordering is the `Greater` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.is_gt(), false);
    /// assert_eq!(Ordering::Equal.is_gt(), false);
    /// assert_eq!(Ordering::Greater.is_gt(), true);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "ordering_helpers", since = "1.53.0")]
    #[stable(feature = "ordering_helpers", since = "1.53.0")]
    pub const fn is_gt(self) -> bool {
        self.as_raw() > 0
    }

    /// Returns `true` if the ordering is either the `Less` or `Equal` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.is_le(), true);
    /// assert_eq!(Ordering::Equal.is_le(), true);
    /// assert_eq!(Ordering::Greater.is_le(), false);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "ordering_helpers", since = "1.53.0")]
    #[stable(feature = "ordering_helpers", since = "1.53.0")]
    pub const fn is_le(self) -> bool {
        self.as_raw() <= 0
    }

    /// Returns `true` if the ordering is either the `Greater` or `Equal` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// assert_eq!(Ordering::Less.is_ge(), false);
    /// assert_eq!(Ordering::Equal.is_ge(), true);
    /// assert_eq!(Ordering::Greater.is_ge(), true);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "ordering_helpers", since = "1.53.0")]
    #[stable(feature = "ordering_helpers", since = "1.53.0")]
    pub const fn is_ge(self) -> bool {
        self.as_raw() >= 0
    }

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
    /// let data: &mut [_] = &mut [2, 10, 5, 8];
    ///
    /// // sort the array from largest to smallest.
    /// data.sort_by(|a, b| a.cmp(b).reverse());
    ///
    /// let b: &mut [_] = &mut [10, 8, 5, 2];
    /// assert!(data == b);
    /// ```
    #[inline]
    #[must_use]
    #[rustc_const_stable(feature = "const_ordering", since = "1.48.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn reverse(self) -> Ordering {
        match self {
            Less => Greater,
            Equal => Equal,
            Greater => Less,
        }
    }

    /// Chains two orderings.
    ///
    /// Returns `self` when it's not `Equal`. Otherwise returns `other`.
    ///
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
    #[must_use]
    #[rustc_const_stable(feature = "const_ordering", since = "1.48.0")]
    #[stable(feature = "ordering_chaining", since = "1.17.0")]
    pub const fn then(self, other: Ordering) -> Ordering {
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
    /// let y: (i64, i64, i64) = (1, 5, 3);
    /// let result = x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)).then_with(|| x.2.cmp(&y.2));
    ///
    /// assert_eq!(result, Ordering::Less);
    /// ```
    #[inline]
    #[must_use]
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
/// This struct is a helper to be used with functions like [`Vec::sort_by_key`] and
/// can be used to reverse order a part of a key.
///
/// [`Vec::sort_by_key`]: ../../std/vec/struct.Vec.html#method.sort_by_key
///
/// # Examples
///
/// ```
/// use std::cmp::Reverse;
///
/// let mut v = vec![1, 2, 3, 4, 5, 6];
/// v.sort_by_key(|&num| (num > 3, Reverse(num)));
/// assert_eq!(v, vec![3, 2, 1, 6, 5, 4]);
/// ```
#[derive(PartialEq, Eq, Debug, Copy, Default, Hash)]
#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
#[repr(transparent)]
pub struct Reverse<T>(#[stable(feature = "reverse_cmp_key", since = "1.19.0")] pub T);

#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
impl<T: PartialOrd> PartialOrd for Reverse<T> {
    #[inline]
    fn partial_cmp(&self, other: &Reverse<T>) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        other.0 < self.0
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        other.0 <= self.0
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        other.0 > self.0
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        other.0 >= self.0
    }
}

#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
impl<T: Ord> Ord for Reverse<T> {
    #[inline]
    fn cmp(&self, other: &Reverse<T>) -> Ordering {
        other.0.cmp(&self.0)
    }
}

#[stable(feature = "reverse_cmp_key", since = "1.19.0")]
impl<T: Clone> Clone for Reverse<T> {
    #[inline]
    fn clone(&self) -> Reverse<T> {
        Reverse(self.0.clone())
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0)
    }
}

/// Trait for types that form a [total order](https://en.wikipedia.org/wiki/Total_order).
///
/// Implementations must be consistent with the [`PartialOrd`] implementation, and ensure `max`,
/// `min`, and `clamp` are consistent with `cmp`:
///
/// - `partial_cmp(a, b) == Some(cmp(a, b))`.
/// - `max(a, b) == max_by(a, b, cmp)` (ensured by the default implementation).
/// - `min(a, b) == min_by(a, b, cmp)` (ensured by the default implementation).
/// - For `a.clamp(min, max)`, see the [method docs](#method.clamp) (ensured by the default
///   implementation).
///
/// Violating these requirements is a logic error. The behavior resulting from a logic error is not
/// specified, but users of the trait must ensure that such logic errors do *not* result in
/// undefined behavior. This means that `unsafe` code **must not** rely on the correctness of these
/// methods.
///
/// ## Corollaries
///
/// From the above and the requirements of `PartialOrd`, it follows that for all `a`, `b` and `c`:
///
/// - exactly one of `a < b`, `a == b` or `a > b` is true; and
/// - `<` is transitive: `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and
///   `>`.
///
/// Mathematically speaking, the `<` operator defines a strict [weak order]. In cases where `==`
/// conforms to mathematical equality, it also defines a strict [total order].
///
/// [weak order]: https://en.wikipedia.org/wiki/Weak_ordering
/// [total order]: https://en.wikipedia.org/wiki/Total_order
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`.
///
/// When `derive`d on structs, it will produce a
/// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order) ordering based on the
/// top-to-bottom declaration order of the struct's members.
///
/// When `derive`d on enums, variants are ordered primarily by their discriminants. Secondarily,
/// they are ordered by their fields. By default, the discriminant is smallest for variants at the
/// top, and largest for variants at the bottom. Here's an example:
///
/// ```
/// #[derive(PartialEq, Eq, PartialOrd, Ord)]
/// enum E {
///     Top,
///     Bottom,
/// }
///
/// assert!(E::Top < E::Bottom);
/// ```
///
/// However, manually setting the discriminants can override this default behavior:
///
/// ```
/// #[derive(PartialEq, Eq, PartialOrd, Ord)]
/// enum E {
///     Top = 2,
///     Bottom = 1,
/// }
///
/// assert!(E::Bottom < E::Top);
/// ```
///
/// ## Lexicographical comparison
///
/// Lexicographical comparison is an operation with the following properties:
///  - Two sequences are compared element by element.
///  - The first mismatching element defines which sequence is lexicographically less or greater
///    than the other.
///  - If one sequence is a prefix of another, the shorter sequence is lexicographically less than
///    the other.
///  - If two sequences have equivalent elements and are of the same length, then the sequences are
///    lexicographically equal.
///  - An empty sequence is lexicographically less than any non-empty sequence.
///  - Two empty sequences are lexicographically equal.
///
/// ## How can I implement `Ord`?
///
/// `Ord` requires that the type also be [`PartialOrd`], [`PartialEq`], and [`Eq`].
///
/// Because `Ord` implies a stronger ordering relationship than [`PartialOrd`], and both `Ord` and
/// [`PartialOrd`] must agree, you must choose how to implement `Ord` **first**. You can choose to
/// derive it, or implement it manually. If you derive it, you should derive all four traits. If you
/// implement it manually, you should manually implement all four traits, based on the
/// implementation of `Ord`.
///
/// Here's an example where you want to define the `Character` comparison by `health` and
/// `experience` only, disregarding the field `mana`:
///
/// ```
/// use std::cmp::Ordering;
///
/// struct Character {
///     health: u32,
///     experience: u32,
///     mana: f32,
/// }
///
/// impl Ord for Character {
///     fn cmp(&self, other: &Self) -> Ordering {
///         self.experience
///             .cmp(&other.experience)
///             .then(self.health.cmp(&other.health))
///     }
/// }
///
/// impl PartialOrd for Character {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// impl PartialEq for Character {
///     fn eq(&self, other: &Self) -> bool {
///         self.health == other.health && self.experience == other.experience
///     }
/// }
///
/// impl Eq for Character {}
/// ```
///
/// If all you need is to `slice::sort` a type by a field value, it can be simpler to use
/// `slice::sort_by_key`.
///
/// ## Examples of incorrect `Ord` implementations
///
/// ```
/// use std::cmp::Ordering;
///
/// #[derive(Debug)]
/// struct Character {
///     health: f32,
/// }
///
/// impl Ord for Character {
///     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
///         if self.health < other.health {
///             Ordering::Less
///         } else if self.health > other.health {
///             Ordering::Greater
///         } else {
///             Ordering::Equal
///         }
///     }
/// }
///
/// impl PartialOrd for Character {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// impl PartialEq for Character {
///     fn eq(&self, other: &Self) -> bool {
///         self.health == other.health
///     }
/// }
///
/// impl Eq for Character {}
///
/// let a = Character { health: 4.5 };
/// let b = Character { health: f32::NAN };
///
/// // Mistake: floating-point values do not form a total order and using the built-in comparison
/// // operands to implement `Ord` irregardless of that reality does not change it. Use
/// // `f32::total_cmp` if you need a total order for floating-point values.
///
/// // Reflexivity requirement of `Ord` is not given.
/// assert!(a == a);
/// assert!(b != b);
///
/// // Antisymmetry requirement of `Ord` is not given. Only one of a < c and c < a is allowed to be
/// // true, not both or neither.
/// assert_eq!((a < b) as u8 + (b < a) as u8, 0);
/// ```
///
/// ```
/// use std::cmp::Ordering;
///
/// #[derive(Debug)]
/// struct Character {
///     health: u32,
///     experience: u32,
/// }
///
/// impl PartialOrd for Character {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// impl Ord for Character {
///     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
///         if self.health < 50 {
///             self.health.cmp(&other.health)
///         } else {
///             self.experience.cmp(&other.experience)
///         }
///     }
/// }
///
/// // For performance reasons implementing `PartialEq` this way is not the idiomatic way, but it
/// // ensures consistent behavior between `PartialEq`, `PartialOrd` and `Ord` in this example.
/// impl PartialEq for Character {
///     fn eq(&self, other: &Self) -> bool {
///         self.cmp(other) == Ordering::Equal
///     }
/// }
///
/// impl Eq for Character {}
///
/// let a = Character {
///     health: 3,
///     experience: 5,
/// };
/// let b = Character {
///     health: 10,
///     experience: 77,
/// };
/// let c = Character {
///     health: 143,
///     experience: 2,
/// };
///
/// // Mistake: The implementation of `Ord` compares different fields depending on the value of
/// // `self.health`, the resulting order is not total.
///
/// // Transitivity requirement of `Ord` is not given. If a is smaller than b and b is smaller than
/// // c, by transitive property a must also be smaller than c.
/// assert!(a < b && b < c && c < a);
///
/// // Antisymmetry requirement of `Ord` is not given. Only one of a < c and c < a is allowed to be
/// // true, not both or neither.
/// assert_eq!((a < c) as u8 + (c < a) as u8, 2);
/// ```
///
/// The documentation of [`PartialOrd`] contains further examples, for example it's wrong for
/// [`PartialOrd`] and [`PartialEq`] to disagree.
///
/// [`cmp`]: Ord::cmp
#[doc(alias = "<")]
#[doc(alias = ">")]
#[doc(alias = "<=")]
#[doc(alias = ">=")]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Ord"]
pub trait Ord: Eq + PartialOrd<Self> {
    /// This method returns an [`Ordering`] between `self` and `other`.
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "ord_cmp_method"]
    fn cmp(&self, other: &Self) -> Ordering;

    /// Compares and returns the maximum of two values.
    ///
    /// Returns the second argument if the comparison determines them to be equal.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1.max(2), 2);
    /// assert_eq!(2.max(2), 2);
    /// ```
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// #[derive(Eq)]
    /// struct Equal(&'static str);
    ///
    /// impl PartialEq for Equal {
    ///     fn eq(&self, other: &Self) -> bool { true }
    /// }
    /// impl PartialOrd for Equal {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(Ordering::Equal) }
    /// }
    /// impl Ord for Equal {
    ///     fn cmp(&self, other: &Self) -> Ordering { Ordering::Equal }
    /// }
    ///
    /// assert_eq!(Equal("self").max(Equal("other")).0, "other");
    /// ```
    #[stable(feature = "ord_max_min", since = "1.21.0")]
    #[inline]
    #[must_use]
    #[rustc_diagnostic_item = "cmp_ord_max"]
    fn max(self, other: Self) -> Self
    where
        Self: Sized,
    {
        if other < self { self } else { other }
    }

    /// Compares and returns the minimum of two values.
    ///
    /// Returns the first argument if the comparison determines them to be equal.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1.min(2), 1);
    /// assert_eq!(2.min(2), 2);
    /// ```
    /// ```
    /// use std::cmp::Ordering;
    ///
    /// #[derive(Eq)]
    /// struct Equal(&'static str);
    ///
    /// impl PartialEq for Equal {
    ///     fn eq(&self, other: &Self) -> bool { true }
    /// }
    /// impl PartialOrd for Equal {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(Ordering::Equal) }
    /// }
    /// impl Ord for Equal {
    ///     fn cmp(&self, other: &Self) -> Ordering { Ordering::Equal }
    /// }
    ///
    /// assert_eq!(Equal("self").min(Equal("other")).0, "self");
    /// ```
    #[stable(feature = "ord_max_min", since = "1.21.0")]
    #[inline]
    #[must_use]
    #[rustc_diagnostic_item = "cmp_ord_min"]
    fn min(self, other: Self) -> Self
    where
        Self: Sized,
    {
        if other < self { other } else { self }
    }

    /// Restrict a value to a certain interval.
    ///
    /// Returns `max` if `self` is greater than `max`, and `min` if `self` is
    /// less than `min`. Otherwise this returns `self`.
    ///
    /// # Panics
    ///
    /// Panics if `min > max`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!((-3).clamp(-2, 1), -2);
    /// assert_eq!(0.clamp(-2, 1), 0);
    /// assert_eq!(2.clamp(-2, 1), 1);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "clamp", since = "1.50.0")]
    fn clamp(self, min: Self, max: Self) -> Self
    where
        Self: Sized,
    {
        assert!(min <= max);
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
}

/// Derive macro generating an impl of the trait [`Ord`].
/// The behavior of this macro is described in detail [here](Ord#derivable).
#[rustc_builtin_macro]
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow_internal_unstable(core_intrinsics)]
pub macro Ord($item:item) {
    /* compiler built-in */
}

/// Trait for types that form a [partial order](https://en.wikipedia.org/wiki/Partial_order).
///
/// The `lt`, `le`, `gt`, and `ge` methods of this trait can be called using the `<`, `<=`, `>`, and
/// `>=` operators, respectively.
///
/// This trait should **only** contain the comparison logic for a type **if one plans on only
/// implementing `PartialOrd` but not [`Ord`]**. Otherwise the comparison logic should be in [`Ord`]
/// and this trait implemented with `Some(self.cmp(other))`.
///
/// The methods of this trait must be consistent with each other and with those of [`PartialEq`].
/// The following conditions must hold:
///
/// 1. `a == b` if and only if `partial_cmp(a, b) == Some(Equal)`.
/// 2. `a < b` if and only if `partial_cmp(a, b) == Some(Less)`
/// 3. `a > b` if and only if `partial_cmp(a, b) == Some(Greater)`
/// 4. `a <= b` if and only if `a < b || a == b`
/// 5. `a >= b` if and only if `a > b || a == b`
/// 6. `a != b` if and only if `!(a == b)`.
///
/// Conditions 2–5 above are ensured by the default implementation. Condition 6 is already ensured
/// by [`PartialEq`].
///
/// If [`Ord`] is also implemented for `Self` and `Rhs`, it must also be consistent with
/// `partial_cmp` (see the documentation of that trait for the exact requirements). It's easy to
/// accidentally make them disagree by deriving some of the traits and manually implementing others.
///
/// The comparison relations must satisfy the following conditions (for all `a`, `b`, `c` of type
/// `A`, `B`, `C`):
///
/// - **Transitivity**: if `A: PartialOrd<B>` and `B: PartialOrd<C>` and `A: PartialOrd<C>`, then `a
///   < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`. This must also
///   work for longer chains, such as when `A: PartialOrd<B>`, `B: PartialOrd<C>`, `C:
///   PartialOrd<D>`, and `A: PartialOrd<D>` all exist.
/// - **Duality**: if `A: PartialOrd<B>` and `B: PartialOrd<A>`, then `a < b` if and only if `b >
///   a`.
///
/// Note that the `B: PartialOrd<A>` (dual) and `A: PartialOrd<C>` (transitive) impls are not forced
/// to exist, but these requirements apply whenever they do exist.
///
/// Violating these requirements is a logic error. The behavior resulting from a logic error is not
/// specified, but users of the trait must ensure that such logic errors do *not* result in
/// undefined behavior. This means that `unsafe` code **must not** rely on the correctness of these
/// methods.
///
/// ## Cross-crate considerations
///
/// Upholding the requirements stated above can become tricky when one crate implements `PartialOrd`
/// for a type of another crate (i.e., to allow comparing one of its own types with a type from the
/// standard library). The recommendation is to never implement this trait for a foreign type. In
/// other words, such a crate should do `impl PartialOrd<ForeignType> for LocalType`, but it should
/// *not* do `impl PartialOrd<LocalType> for ForeignType`.
///
/// This avoids the problem of transitive chains that criss-cross crate boundaries: for all local
/// types `T`, you may assume that no other crate will add `impl`s that allow comparing `T < U`. In
/// other words, if other crates add `impl`s that allow building longer transitive chains `U1 < ...
/// < T < V1 < ...`, then all the types that appear to the right of `T` must be types that the crate
/// defining `T` already knows about. This rules out transitive chains where downstream crates can
/// add new `impl`s that "stitch together" comparisons of foreign types in ways that violate
/// transitivity.
///
/// Not having such foreign `impl`s also avoids forward compatibility issues where one crate adding
/// more `PartialOrd` implementations can cause build failures in downstream crates.
///
/// ## Corollaries
///
/// The following corollaries follow from the above requirements:
///
/// - irreflexivity of `<` and `>`: `!(a < a)`, `!(a > a)`
/// - transitivity of `>`: if `a > b` and `b > c` then `a > c`
/// - duality of `partial_cmp`: `partial_cmp(a, b) == partial_cmp(b, a).map(Ordering::reverse)`
///
/// ## Strict and non-strict partial orders
///
/// The `<` and `>` operators behave according to a *strict* partial order. However, `<=` and `>=`
/// do **not** behave according to a *non-strict* partial order. That is because mathematically, a
/// non-strict partial order would require reflexivity, i.e. `a <= a` would need to be true for
/// every `a`. This isn't always the case for types that implement `PartialOrd`, for example:
///
/// ```
/// let a = f64::sqrt(-1.0);
/// assert_eq!(a <= a, false);
/// ```
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`.
///
/// When `derive`d on structs, it will produce a
/// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order) ordering based on the
/// top-to-bottom declaration order of the struct's members.
///
/// When `derive`d on enums, variants are primarily ordered by their discriminants. Secondarily,
/// they are ordered by their fields. By default, the discriminant is smallest for variants at the
/// top, and largest for variants at the bottom. Here's an example:
///
/// ```
/// #[derive(PartialEq, PartialOrd)]
/// enum E {
///     Top,
///     Bottom,
/// }
///
/// assert!(E::Top < E::Bottom);
/// ```
///
/// However, manually setting the discriminants can override this default behavior:
///
/// ```
/// #[derive(PartialEq, PartialOrd)]
/// enum E {
///     Top = 2,
///     Bottom = 1,
/// }
///
/// assert!(E::Bottom < E::Top);
/// ```
///
/// ## How can I implement `PartialOrd`?
///
/// `PartialOrd` only requires implementation of the [`partial_cmp`] method, with the others
/// generated from default implementations.
///
/// However it remains possible to implement the others separately for types which do not have a
/// total order. For example, for floating point numbers, `NaN < 0 == false` and `NaN >= 0 == false`
/// (cf. IEEE 754-2008 section 5.11).
///
/// `PartialOrd` requires your type to be [`PartialEq`].
///
/// If your type is [`Ord`], you can implement [`partial_cmp`] by using [`cmp`]:
///
/// ```
/// use std::cmp::Ordering;
///
/// struct Person {
///     id: u32,
///     name: String,
///     height: u32,
/// }
///
/// impl PartialOrd for Person {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         Some(self.cmp(other))
///     }
/// }
///
/// impl Ord for Person {
///     fn cmp(&self, other: &Self) -> Ordering {
///         self.height.cmp(&other.height)
///     }
/// }
///
/// impl PartialEq for Person {
///     fn eq(&self, other: &Self) -> bool {
///         self.height == other.height
///     }
/// }
///
/// impl Eq for Person {}
/// ```
///
/// You may also find it useful to use [`partial_cmp`] on your type's fields. Here is an example of
/// `Person` types who have a floating-point `height` field that is the only field to be used for
/// sorting:
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
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         self.height.partial_cmp(&other.height)
///     }
/// }
///
/// impl PartialEq for Person {
///     fn eq(&self, other: &Self) -> bool {
///         self.height == other.height
///     }
/// }
/// ```
///
/// ## Examples of incorrect `PartialOrd` implementations
///
/// ```
/// use std::cmp::Ordering;
///
/// #[derive(PartialEq, Debug)]
/// struct Character {
///     health: u32,
///     experience: u32,
/// }
///
/// impl PartialOrd for Character {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         Some(self.health.cmp(&other.health))
///     }
/// }
///
/// let a = Character {
///     health: 10,
///     experience: 5,
/// };
/// let b = Character {
///     health: 10,
///     experience: 77,
/// };
///
/// // Mistake: `PartialEq` and `PartialOrd` disagree with each other.
///
/// assert_eq!(a.partial_cmp(&b).unwrap(), Ordering::Equal); // a == b according to `PartialOrd`.
/// assert_ne!(a, b); // a != b according to `PartialEq`.
/// ```
///
/// # Examples
///
/// ```
/// let x: u32 = 0;
/// let y: u32 = 1;
///
/// assert_eq!(x < y, true);
/// assert_eq!(x.lt(&y), true);
/// ```
///
/// [`partial_cmp`]: PartialOrd::partial_cmp
/// [`cmp`]: Ord::cmp
#[lang = "partial_ord"]
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(alias = ">")]
#[doc(alias = "<")]
#[doc(alias = "<=")]
#[doc(alias = ">=")]
#[rustc_on_unimplemented(
    message = "can't compare `{Self}` with `{Rhs}`",
    label = "no implementation for `{Self} < {Rhs}` and `{Self} > {Rhs}`",
    append_const_msg
)]
#[rustc_diagnostic_item = "PartialOrd"]
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
    /// let result = f64::NAN.partial_cmp(&1.0);
    /// assert_eq!(result, None);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialord_cmp"]
    fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;

    /// Tests less than (for `self` and `other`) and is used by the `<` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1.0 < 1.0, false);
    /// assert_eq!(1.0 < 2.0, true);
    /// assert_eq!(2.0 < 1.0, false);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialord_lt"]
    fn lt(&self, other: &Rhs) -> bool {
        self.partial_cmp(other).is_some_and(Ordering::is_lt)
    }

    /// Tests less than or equal to (for `self` and `other`) and is used by the
    /// `<=` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1.0 <= 1.0, true);
    /// assert_eq!(1.0 <= 2.0, true);
    /// assert_eq!(2.0 <= 1.0, false);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialord_le"]
    fn le(&self, other: &Rhs) -> bool {
        self.partial_cmp(other).is_some_and(Ordering::is_le)
    }

    /// Tests greater than (for `self` and `other`) and is used by the `>`
    /// operator.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1.0 > 1.0, false);
    /// assert_eq!(1.0 > 2.0, false);
    /// assert_eq!(2.0 > 1.0, true);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialord_gt"]
    fn gt(&self, other: &Rhs) -> bool {
        self.partial_cmp(other).is_some_and(Ordering::is_gt)
    }

    /// Tests greater than or equal to (for `self` and `other`) and is used by
    /// the `>=` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(1.0 >= 1.0, true);
    /// assert_eq!(1.0 >= 2.0, false);
    /// assert_eq!(2.0 >= 1.0, true);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_diagnostic_item = "cmp_partialord_ge"]
    fn ge(&self, other: &Rhs) -> bool {
        self.partial_cmp(other).is_some_and(Ordering::is_ge)
    }

    /// If `self == other`, returns `ControlFlow::Continue(())`.
    /// Otherwise, returns `ControlFlow::Break(self < other)`.
    ///
    /// This is useful for chaining together calls when implementing a lexical
    /// `PartialOrd::lt`, as it allows types (like primitives) which can cheaply
    /// check `==` and `<` separately to do rather than needing to calculate
    /// (then optimize out) the three-way `Ordering` result.
    #[inline]
    #[must_use]
    // Added to improve the behaviour of tuples; not necessarily stabilization-track.
    #[unstable(feature = "partial_ord_chaining_methods", issue = "none")]
    #[doc(hidden)]
    fn __chaining_lt(&self, other: &Rhs) -> ControlFlow<bool> {
        default_chaining_impl(self, other, Ordering::is_lt)
    }

    /// Same as `__chaining_lt`, but for `<=` instead of `<`.
    #[inline]
    #[must_use]
    #[unstable(feature = "partial_ord_chaining_methods", issue = "none")]
    #[doc(hidden)]
    fn __chaining_le(&self, other: &Rhs) -> ControlFlow<bool> {
        default_chaining_impl(self, other, Ordering::is_le)
    }

    /// Same as `__chaining_lt`, but for `>` instead of `<`.
    #[inline]
    #[must_use]
    #[unstable(feature = "partial_ord_chaining_methods", issue = "none")]
    #[doc(hidden)]
    fn __chaining_gt(&self, other: &Rhs) -> ControlFlow<bool> {
        default_chaining_impl(self, other, Ordering::is_gt)
    }

    /// Same as `__chaining_lt`, but for `>=` instead of `<`.
    #[inline]
    #[must_use]
    #[unstable(feature = "partial_ord_chaining_methods", issue = "none")]
    #[doc(hidden)]
    fn __chaining_ge(&self, other: &Rhs) -> ControlFlow<bool> {
        default_chaining_impl(self, other, Ordering::is_ge)
    }
}

fn default_chaining_impl<T: ?Sized, U: ?Sized>(
    lhs: &T,
    rhs: &U,
    p: impl FnOnce(Ordering) -> bool,
) -> ControlFlow<bool>
where
    T: PartialOrd<U>,
{
    // It's important that this only call `partial_cmp` once, not call `eq` then
    // one of the relational operators.  We don't want to `bcmp`-then-`memcp` a
    // `String`, for example, or similarly for other data structures (#108157).
    match <T as PartialOrd<U>>::partial_cmp(lhs, rhs) {
        Some(Equal) => ControlFlow::Continue(()),
        Some(c) => ControlFlow::Break(p(c)),
        None => ControlFlow::Break(false),
    }
}

/// Derive macro generating an impl of the trait [`PartialOrd`].
/// The behavior of this macro is described in detail [here](PartialOrd#derivable).
#[rustc_builtin_macro]
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow_internal_unstable(core_intrinsics)]
pub macro PartialOrd($item:item) {
    /* compiler built-in */
}

/// Compares and returns the minimum of two values.
///
/// Returns the first argument if the comparison determines them to be equal.
///
/// Internally uses an alias to [`Ord::min`].
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// assert_eq!(cmp::min(1, 2), 1);
/// assert_eq!(cmp::min(2, 2), 2);
/// ```
/// ```
/// use std::cmp::{self, Ordering};
///
/// #[derive(Eq)]
/// struct Equal(&'static str);
///
/// impl PartialEq for Equal {
///     fn eq(&self, other: &Self) -> bool { true }
/// }
/// impl PartialOrd for Equal {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(Ordering::Equal) }
/// }
/// impl Ord for Equal {
///     fn cmp(&self, other: &Self) -> Ordering { Ordering::Equal }
/// }
///
/// assert_eq!(cmp::min(Equal("v1"), Equal("v2")).0, "v1");
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "cmp_min"]
pub fn min<T: Ord>(v1: T, v2: T) -> T {
    v1.min(v2)
}

/// Returns the minimum of two values with respect to the specified comparison function.
///
/// Returns the first argument if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// let abs_cmp = |x: &i32, y: &i32| x.abs().cmp(&y.abs());
///
/// let result = cmp::min_by(2, -1, abs_cmp);
/// assert_eq!(result, -1);
///
/// let result = cmp::min_by(2, -3, abs_cmp);
/// assert_eq!(result, 2);
///
/// let result = cmp::min_by(1, -1, abs_cmp);
/// assert_eq!(result, 1);
/// ```
#[inline]
#[must_use]
#[stable(feature = "cmp_min_max_by", since = "1.53.0")]
pub fn min_by<T, F: FnOnce(&T, &T) -> Ordering>(v1: T, v2: T, compare: F) -> T {
    if compare(&v2, &v1).is_lt() { v2 } else { v1 }
}

/// Returns the element that gives the minimum value from the specified function.
///
/// Returns the first argument if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// let result = cmp::min_by_key(2, -1, |x: &i32| x.abs());
/// assert_eq!(result, -1);
///
/// let result = cmp::min_by_key(2, -3, |x: &i32| x.abs());
/// assert_eq!(result, 2);
///
/// let result = cmp::min_by_key(1, -1, |x: &i32| x.abs());
/// assert_eq!(result, 1);
/// ```
#[inline]
#[must_use]
#[stable(feature = "cmp_min_max_by", since = "1.53.0")]
pub fn min_by_key<T, F: FnMut(&T) -> K, K: Ord>(v1: T, v2: T, mut f: F) -> T {
    if f(&v2) < f(&v1) { v2 } else { v1 }
}

/// Compares and returns the maximum of two values.
///
/// Returns the second argument if the comparison determines them to be equal.
///
/// Internally uses an alias to [`Ord::max`].
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// assert_eq!(cmp::max(1, 2), 2);
/// assert_eq!(cmp::max(2, 2), 2);
/// ```
/// ```
/// use std::cmp::{self, Ordering};
///
/// #[derive(Eq)]
/// struct Equal(&'static str);
///
/// impl PartialEq for Equal {
///     fn eq(&self, other: &Self) -> bool { true }
/// }
/// impl PartialOrd for Equal {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(Ordering::Equal) }
/// }
/// impl Ord for Equal {
///     fn cmp(&self, other: &Self) -> Ordering { Ordering::Equal }
/// }
///
/// assert_eq!(cmp::max(Equal("v1"), Equal("v2")).0, "v2");
/// ```
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "cmp_max"]
pub fn max<T: Ord>(v1: T, v2: T) -> T {
    v1.max(v2)
}

/// Returns the maximum of two values with respect to the specified comparison function.
///
/// Returns the second argument if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// let abs_cmp = |x: &i32, y: &i32| x.abs().cmp(&y.abs());
///
/// let result = cmp::max_by(3, -2, abs_cmp) ;
/// assert_eq!(result, 3);
///
/// let result = cmp::max_by(1, -2, abs_cmp);
/// assert_eq!(result, -2);
///
/// let result = cmp::max_by(1, -1, abs_cmp);
/// assert_eq!(result, -1);
/// ```
#[inline]
#[must_use]
#[stable(feature = "cmp_min_max_by", since = "1.53.0")]
pub fn max_by<T, F: FnOnce(&T, &T) -> Ordering>(v1: T, v2: T, compare: F) -> T {
    if compare(&v2, &v1).is_lt() { v1 } else { v2 }
}

/// Returns the element that gives the maximum value from the specified function.
///
/// Returns the second argument if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// use std::cmp;
///
/// let result = cmp::max_by_key(3, -2, |x: &i32| x.abs());
/// assert_eq!(result, 3);
///
/// let result = cmp::max_by_key(1, -2, |x: &i32| x.abs());
/// assert_eq!(result, -2);
///
/// let result = cmp::max_by_key(1, -1, |x: &i32| x.abs());
/// assert_eq!(result, -1);
/// ```
#[inline]
#[must_use]
#[stable(feature = "cmp_min_max_by", since = "1.53.0")]
pub fn max_by_key<T, F: FnMut(&T) -> K, K: Ord>(v1: T, v2: T, mut f: F) -> T {
    if f(&v2) < f(&v1) { v1 } else { v2 }
}

/// Compares and sorts two values, returning minimum and maximum.
///
/// Returns `[v1, v2]` if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// #![feature(cmp_minmax)]
/// use std::cmp;
///
/// assert_eq!(cmp::minmax(1, 2), [1, 2]);
/// assert_eq!(cmp::minmax(2, 1), [1, 2]);
///
/// // You can destructure the result using array patterns
/// let [min, max] = cmp::minmax(42, 17);
/// assert_eq!(min, 17);
/// assert_eq!(max, 42);
/// ```
/// ```
/// #![feature(cmp_minmax)]
/// use std::cmp::{self, Ordering};
///
/// #[derive(Eq)]
/// struct Equal(&'static str);
///
/// impl PartialEq for Equal {
///     fn eq(&self, other: &Self) -> bool { true }
/// }
/// impl PartialOrd for Equal {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(Ordering::Equal) }
/// }
/// impl Ord for Equal {
///     fn cmp(&self, other: &Self) -> Ordering { Ordering::Equal }
/// }
///
/// assert_eq!(cmp::minmax(Equal("v1"), Equal("v2")).map(|v| v.0), ["v1", "v2"]);
/// ```
#[inline]
#[must_use]
#[unstable(feature = "cmp_minmax", issue = "115939")]
pub fn minmax<T>(v1: T, v2: T) -> [T; 2]
where
    T: Ord,
{
    if v2 < v1 { [v2, v1] } else { [v1, v2] }
}

/// Returns minimum and maximum values with respect to the specified comparison function.
///
/// Returns `[v1, v2]` if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// #![feature(cmp_minmax)]
/// use std::cmp;
///
/// let abs_cmp = |x: &i32, y: &i32| x.abs().cmp(&y.abs());
///
/// assert_eq!(cmp::minmax_by(-2, 1, abs_cmp), [1, -2]);
/// assert_eq!(cmp::minmax_by(-1, 2, abs_cmp), [-1, 2]);
/// assert_eq!(cmp::minmax_by(-2, 2, abs_cmp), [-2, 2]);
///
/// // You can destructure the result using array patterns
/// let [min, max] = cmp::minmax_by(-42, 17, abs_cmp);
/// assert_eq!(min, 17);
/// assert_eq!(max, -42);
/// ```
#[inline]
#[must_use]
#[unstable(feature = "cmp_minmax", issue = "115939")]
pub fn minmax_by<T, F>(v1: T, v2: T, compare: F) -> [T; 2]
where
    F: FnOnce(&T, &T) -> Ordering,
{
    if compare(&v2, &v1).is_lt() { [v2, v1] } else { [v1, v2] }
}

/// Returns minimum and maximum values with respect to the specified key function.
///
/// Returns `[v1, v2]` if the comparison determines them to be equal.
///
/// # Examples
///
/// ```
/// #![feature(cmp_minmax)]
/// use std::cmp;
///
/// assert_eq!(cmp::minmax_by_key(-2, 1, |x: &i32| x.abs()), [1, -2]);
/// assert_eq!(cmp::minmax_by_key(-2, 2, |x: &i32| x.abs()), [-2, 2]);
///
/// // You can destructure the result using array patterns
/// let [min, max] = cmp::minmax_by_key(-42, 17, |x: &i32| x.abs());
/// assert_eq!(min, 17);
/// assert_eq!(max, -42);
/// ```
#[inline]
#[must_use]
#[unstable(feature = "cmp_minmax", issue = "115939")]
pub fn minmax_by_key<T, F, K>(v1: T, v2: T, mut f: F) -> [T; 2]
where
    F: FnMut(&T) -> K,
    K: Ord,
{
    if f(&v2) < f(&v1) { [v2, v1] } else { [v1, v2] }
}

// Implementation of PartialEq, Eq, PartialOrd and Ord for primitive types
mod impls {
    use crate::cmp::Ordering::{self, Equal, Greater, Less};
    use crate::hint::unreachable_unchecked;
    use crate::ops::ControlFlow::{self, Break, Continue};

    macro_rules! partial_eq_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl PartialEq for $t {
                #[inline]
                fn eq(&self, other: &Self) -> bool { *self == *other }
                #[inline]
                fn ne(&self, other: &Self) -> bool { *self != *other }
            }
        )*)
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialEq for () {
        #[inline]
        fn eq(&self, _other: &()) -> bool {
            true
        }
        #[inline]
        fn ne(&self, _other: &()) -> bool {
            false
        }
    }

    partial_eq_impl! {
        bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f16 f32 f64 f128
    }

    macro_rules! eq_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Eq for $t {}
        )*)
    }

    eq_impl! { () bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

    #[rustfmt::skip]
    macro_rules! partial_ord_methods_primitive_impl {
        () => {
            #[inline(always)]
            fn lt(&self, other: &Self) -> bool { *self <  *other }
            #[inline(always)]
            fn le(&self, other: &Self) -> bool { *self <= *other }
            #[inline(always)]
            fn gt(&self, other: &Self) -> bool { *self >  *other }
            #[inline(always)]
            fn ge(&self, other: &Self) -> bool { *self >= *other }

            // These implementations are the same for `Ord` or `PartialOrd` types
            // because if either is NAN the `==` test will fail so we end up in
            // the `Break` case and the comparison will correctly return `false`.

            #[inline]
            fn __chaining_lt(&self, other: &Self) -> ControlFlow<bool> {
                let (lhs, rhs) = (*self, *other);
                if lhs == rhs { Continue(()) } else { Break(lhs < rhs) }
            }
            #[inline]
            fn __chaining_le(&self, other: &Self) -> ControlFlow<bool> {
                let (lhs, rhs) = (*self, *other);
                if lhs == rhs { Continue(()) } else { Break(lhs <= rhs) }
            }
            #[inline]
            fn __chaining_gt(&self, other: &Self) -> ControlFlow<bool> {
                let (lhs, rhs) = (*self, *other);
                if lhs == rhs { Continue(()) } else { Break(lhs > rhs) }
            }
            #[inline]
            fn __chaining_ge(&self, other: &Self) -> ControlFlow<bool> {
                let (lhs, rhs) = (*self, *other);
                if lhs == rhs { Continue(()) } else { Break(lhs >= rhs) }
            }
        };
    }

    macro_rules! partial_ord_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl PartialOrd for $t {
                #[inline]
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    match (*self <= *other, *self >= *other) {
                        (false, false) => None,
                        (false, true) => Some(Greater),
                        (true, false) => Some(Less),
                        (true, true) => Some(Equal),
                    }
                }

                partial_ord_methods_primitive_impl!();
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
            Some(self.cmp(other))
        }

        partial_ord_methods_primitive_impl!();
    }

    partial_ord_impl! { f16 f32 f64 f128 }

    macro_rules! ord_impl {
        ($($t:ty)*) => ($(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl PartialOrd for $t {
                #[inline]
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(crate::intrinsics::three_way_compare(*self, *other))
                }

                partial_ord_methods_primitive_impl!();
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl Ord for $t {
                #[inline]
                fn cmp(&self, other: &Self) -> Ordering {
                    crate::intrinsics::three_way_compare(*self, *other)
                }
            }
        )*)
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Ord for () {
        #[inline]
        fn cmp(&self, _other: &()) -> Ordering {
            Equal
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Ord for bool {
        #[inline]
        fn cmp(&self, other: &bool) -> Ordering {
            // Casting to i8's and converting the difference to an Ordering generates
            // more optimal assembly.
            // See <https://github.com/rust-lang/rust/issues/66780> for more info.
            match (*self as i8) - (*other as i8) {
                -1 => Less,
                0 => Equal,
                1 => Greater,
                // SAFETY: bool as i8 returns 0 or 1, so the difference can't be anything else
                _ => unsafe { unreachable_unchecked() },
            }
        }

        #[inline]
        fn min(self, other: bool) -> bool {
            self & other
        }

        #[inline]
        fn max(self, other: bool) -> bool {
            self | other
        }

        #[inline]
        fn clamp(self, min: bool, max: bool) -> bool {
            assert!(min <= max);
            self.max(min).min(max)
        }
    }

    ord_impl! { char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

    #[unstable(feature = "never_type", issue = "35121")]
    impl PartialEq for ! {
        #[inline]
        fn eq(&self, _: &!) -> bool {
            *self
        }
    }

    #[unstable(feature = "never_type", issue = "35121")]
    impl Eq for ! {}

    #[unstable(feature = "never_type", issue = "35121")]
    impl PartialOrd for ! {
        #[inline]
        fn partial_cmp(&self, _: &!) -> Option<Ordering> {
            *self
        }
    }

    #[unstable(feature = "never_type", issue = "35121")]
    impl Ord for ! {
        #[inline]
        fn cmp(&self, _: &!) -> Ordering {
            *self
        }
    }

    // & pointers

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized, B: ?Sized> PartialEq<&B> for &A
    where
        A: PartialEq<B>,
    {
        #[inline]
        fn eq(&self, other: &&B) -> bool {
            PartialEq::eq(*self, *other)
        }
        #[inline]
        fn ne(&self, other: &&B) -> bool {
            PartialEq::ne(*self, *other)
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized, B: ?Sized> PartialOrd<&B> for &A
    where
        A: PartialOrd<B>,
    {
        #[inline]
        fn partial_cmp(&self, other: &&B) -> Option<Ordering> {
            PartialOrd::partial_cmp(*self, *other)
        }
        #[inline]
        fn lt(&self, other: &&B) -> bool {
            PartialOrd::lt(*self, *other)
        }
        #[inline]
        fn le(&self, other: &&B) -> bool {
            PartialOrd::le(*self, *other)
        }
        #[inline]
        fn gt(&self, other: &&B) -> bool {
            PartialOrd::gt(*self, *other)
        }
        #[inline]
        fn ge(&self, other: &&B) -> bool {
            PartialOrd::ge(*self, *other)
        }
        #[inline]
        fn __chaining_lt(&self, other: &&B) -> ControlFlow<bool> {
            PartialOrd::__chaining_lt(*self, *other)
        }
        #[inline]
        fn __chaining_le(&self, other: &&B) -> ControlFlow<bool> {
            PartialOrd::__chaining_le(*self, *other)
        }
        #[inline]
        fn __chaining_gt(&self, other: &&B) -> ControlFlow<bool> {
            PartialOrd::__chaining_gt(*self, *other)
        }
        #[inline]
        fn __chaining_ge(&self, other: &&B) -> ControlFlow<bool> {
            PartialOrd::__chaining_ge(*self, *other)
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized> Ord for &A
    where
        A: Ord,
    {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            Ord::cmp(*self, *other)
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized> Eq for &A where A: Eq {}

    // &mut pointers

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized, B: ?Sized> PartialEq<&mut B> for &mut A
    where
        A: PartialEq<B>,
    {
        #[inline]
        fn eq(&self, other: &&mut B) -> bool {
            PartialEq::eq(*self, *other)
        }
        #[inline]
        fn ne(&self, other: &&mut B) -> bool {
            PartialEq::ne(*self, *other)
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized, B: ?Sized> PartialOrd<&mut B> for &mut A
    where
        A: PartialOrd<B>,
    {
        #[inline]
        fn partial_cmp(&self, other: &&mut B) -> Option<Ordering> {
            PartialOrd::partial_cmp(*self, *other)
        }
        #[inline]
        fn lt(&self, other: &&mut B) -> bool {
            PartialOrd::lt(*self, *other)
        }
        #[inline]
        fn le(&self, other: &&mut B) -> bool {
            PartialOrd::le(*self, *other)
        }
        #[inline]
        fn gt(&self, other: &&mut B) -> bool {
            PartialOrd::gt(*self, *other)
        }
        #[inline]
        fn ge(&self, other: &&mut B) -> bool {
            PartialOrd::ge(*self, *other)
        }
        #[inline]
        fn __chaining_lt(&self, other: &&mut B) -> ControlFlow<bool> {
            PartialOrd::__chaining_lt(*self, *other)
        }
        #[inline]
        fn __chaining_le(&self, other: &&mut B) -> ControlFlow<bool> {
            PartialOrd::__chaining_le(*self, *other)
        }
        #[inline]
        fn __chaining_gt(&self, other: &&mut B) -> ControlFlow<bool> {
            PartialOrd::__chaining_gt(*self, *other)
        }
        #[inline]
        fn __chaining_ge(&self, other: &&mut B) -> ControlFlow<bool> {
            PartialOrd::__chaining_ge(*self, *other)
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized> Ord for &mut A
    where
        A: Ord,
    {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            Ord::cmp(*self, *other)
        }
    }
    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized> Eq for &mut A where A: Eq {}

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized, B: ?Sized> PartialEq<&mut B> for &A
    where
        A: PartialEq<B>,
    {
        #[inline]
        fn eq(&self, other: &&mut B) -> bool {
            PartialEq::eq(*self, *other)
        }
        #[inline]
        fn ne(&self, other: &&mut B) -> bool {
            PartialEq::ne(*self, *other)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<A: ?Sized, B: ?Sized> PartialEq<&B> for &mut A
    where
        A: PartialEq<B>,
    {
        #[inline]
        fn eq(&self, other: &&B) -> bool {
            PartialEq::eq(*self, *other)
        }
        #[inline]
        fn ne(&self, other: &&B) -> bool {
            PartialEq::ne(*self, *other)
        }
    }
}
