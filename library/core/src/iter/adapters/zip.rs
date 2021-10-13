use crate::cmp;
use crate::fmt::{self, Debug};
use crate::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, Iterator};
use crate::iter::{InPlaceIterable, SourceIter, TrustedLen};

/// An iterator that iterates two other iterators simultaneously.
///
/// This `struct` is created by [`zip`] or [`Iterator::zip`].
/// See their documentation for more.
#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Zip<A, B> {
    a: A,
    b: B,
}
impl<A: Iterator, B: Iterator> Zip<A, B> {
    pub(in crate::iter) fn new(a: A, b: B) -> Zip<A, B> {
        ZipImpl::new(a, b)
    }
    fn super_nth(&mut self, mut n: usize) -> Option<(A::Item, B::Item)> {
        while let Some(x) = Iterator::next(self) {
            if n == 0 {
                return Some(x);
            }
            n -= 1;
        }
        None
    }
}

/// Converts the arguments to iterators and zips them.
///
/// See the documentation of [`Iterator::zip`] for more.
///
/// # Examples
///
/// ```
/// #![feature(iter_zip)]
/// use std::iter::zip;
///
/// let xs = [1, 2, 3];
/// let ys = [4, 5, 6];
/// for (x, y) in zip(&xs, &ys) {
///     println!("x:{}, y:{}", x, y);
/// }
///
/// // Nested zips are also possible:
/// let zs = [7, 8, 9];
/// for ((x, y), z) in zip(zip(&xs, &ys), &zs) {
///     println!("x:{}, y:{}, z:{}", x, y, z);
/// }
/// ```
#[unstable(feature = "iter_zip", issue = "83574")]
pub fn zip<A, B>(a: A, b: B) -> Zip<A::IntoIter, B::IntoIter>
where
    A: IntoIterator,
    B: IntoIterator,
{
    ZipImpl::new(a.into_iter(), b.into_iter())
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
{
    type Item = (A::Item, B::Item);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        ZipImpl::next(self)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        ZipImpl::size_hint(self)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        ZipImpl::nth(self, n)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        let ((lower_a, _), (lower_b, _)) = (self.a.size_hint(), self.b.size_hint());
        let lower = n.min(lower_a).min(lower_b);
        let batched = match (self.b.advance_by(lower), self.a.advance_by(lower)) {
            (Ok(()), Ok(())) => lower,
            _ => panic!("size_hint contract violation"),
        };

        for i in batched..n {
            if let (_, None) | (None, _) = (self.b.next(), self.a.next()) {
                return Err(i);
            }
        }

        Ok(())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Zip<A, B>
where
    A: DoubleEndedIterator + ExactSizeIterator,
    B: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(A::Item, B::Item)> {
        ZipImpl::next_back(self)
    }
}

// Zip specialization trait
#[doc(hidden)]
trait ZipImpl<A, B> {
    type Item;
    fn new(a: A, b: B) -> Self;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
    fn nth(&mut self, n: usize) -> Option<Self::Item>;
    fn next_back(&mut self) -> Option<Self::Item>
    where
        A: DoubleEndedIterator + ExactSizeIterator,
        B: DoubleEndedIterator + ExactSizeIterator;
}

// Work around limitations of specialization, requiring `default` impls to be repeated
// in intermediary impls.
macro_rules! zip_impl_general_defaults {
    () => {
        default fn new(a: A, b: B) -> Self {
            Zip { a, b }
        }

        #[inline]
        default fn next(&mut self) -> Option<(A::Item, B::Item)> {
            let y = self.b.next()?;
            let x = self.a.next()?;
            Some((x, y))
        }

        #[inline]
        default fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.super_nth(n)
        }

        #[inline]
        default fn next_back(&mut self) -> Option<(A::Item, B::Item)>
        where
            A: DoubleEndedIterator + ExactSizeIterator,
            B: DoubleEndedIterator + ExactSizeIterator,
        {
            // The function body below only uses `self.a/b.len()` and `self.a/b.next_back()`
            // and doesn’t call `next_back` too often, so this implementation is safe in
            // the `TrustedRandomAccessNoCoerce` specialization

            let a_sz = self.a.len();
            let b_sz = self.b.len();
            if a_sz != b_sz {
                // Adjust a, b to equal length
                if a_sz > b_sz {
                    for _ in 0..a_sz - b_sz {
                        self.a.next_back();
                    }
                } else {
                    for _ in 0..b_sz - a_sz {
                        self.b.next_back();
                    }
                }
            }
            match (self.b.next_back(), self.a.next_back()) {
                (Some(y), Some(x)) => Some((x, y)),
                (None, None) => None,
                _ => unreachable!(),
            }
        }
    };
}

// General Zip impl
#[doc(hidden)]
impl<A, B> ZipImpl<A, B> for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
{
    type Item = (A::Item, B::Item);

    zip_impl_general_defaults! {}

    #[inline]
    default fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = cmp::min(a_lower, b_lower);

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) => Some(cmp::min(x, y)),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        };

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> ExactSizeIterator for Zip<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator,
{
}

#[stable(feature = "fused", since = "1.26.0")]
impl<A, B> FusedIterator for Zip<A, B>
where
    A: FusedIterator,
    B: FusedIterator,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A, B> TrustedLen for Zip<A, B>
where
    A: TrustedLen,
    B: TrustedLen,
{
}

// Arbitrarily selects the left side of the zip iteration as extractable "source"
// it would require negative trait bounds to be able to try both
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<A, B> SourceIter for Zip<A, B>
where
    A: SourceIter,
{
    type Source = A::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut A::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.a) }
    }
}

// Since SourceIter forwards the left hand side we do the same here
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<A: InPlaceIterable, B: Iterator> InPlaceIterable for Zip<A, B> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Debug, B: Debug> Debug for Zip<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ZipFmt::fmt(self, f)
    }
}

trait ZipFmt<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<A: Debug, B: Debug> ZipFmt<A, B> for Zip<A, B> {
    default fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Zip").field("a", &self.a).field("b", &self.b).finish()
    }
}

/// An iterator whose items are random-accessible efficiently
///
/// # Safety
///
/// The iterator's `size_hint` must be exact and cheap to call.
///
/// `TrustedRandomAccessNoCoerce::size` may not be overridden.
///
/// All subtypes and all supertypes of `Self` must also implement `TrustedRandomAccess`.
/// In particular, this means that types with non-invariant parameters usually can not have
/// an impl for `TrustedRandomAccess` that depends on any trait bounds on such parameters, except
/// for bounds that come from the respective struct/enum definition itself, or bounds involving
/// traits that themselves come with a guarantee similar to this one.
///
/// If `Self: ExactSizeIterator` then `self.len()` must always produce results consistent
/// with `self.size()`.
///
/// If `Self: Iterator`, then `<Self as Iterator>::__iterator_get_unchecked(&mut self, idx)`
/// must be safe to call provided the following conditions are met.
///
/// 1. `0 <= idx` and `idx < self.size()`.
/// 2. If `Self: !Clone`, then `self.__iterator_get_unchecked(idx)` is never called with the same
///    index on `self` more than once.
/// 3. After `self.__iterator_get_unchecked(idx)` has been called, then `self.next_back()` will
///    only be called at most `self.size() - idx - 1` times. If `Self: Clone` and `self` is cloned,
///    then this number is calculated for `self` and its clone individually,
///    but `self.next_back()` calls that happened before the cloning count for both `self` and the clone.
/// 4. After `self.__iterator_get_unchecked(idx)` has been called, then only the following methods
///    will be called on `self` or on any new clones of `self`:
///     * `std::clone::Clone::clone`
///     * `std::iter::Iterator::size_hint`
///     * `std::iter::DoubleEndedIterator::next_back`
///     * `std::iter::ExactSizeIterator::len`
///     * `std::iter::Iterator::__iterator_get_unchecked`
///     * `std::iter::TrustedRandomAccessNoCoerce::size`
/// 5. If `T` is a subtype of `Self`, then `self` is allowed to be coerced
///    to `T`. If `self` is coerced to `T` after `self.__iterator_get_unchecked(idx)` has already
///    been called, then no methods except for the ones listed under 4. are allowed to be called
///    on the resulting value of type `T`, either. Multiple such coercion steps are allowed.
///    Regarding 2. and 3., the number of times `__iterator_get_unchecked(idx)` or `next_back()` is
///    called on `self` and the resulting value of type `T` (and on further coercion results with
///    sub-subtypes) are added together and their sums must not exceed the specified bounds.
///
/// Further, given that these conditions are met, it must guarantee that:
///
/// * It does not change the value returned from `size_hint`
/// * It must be safe to call the methods listed above on `self` after calling
///   `self.__iterator_get_unchecked(idx)`, assuming that the required traits are implemented.
/// * It must also be safe to drop `self` after calling `self.__iterator_get_unchecked(idx)`.
/// * If `T` is a subtype of `Self`, then it must be safe to coerce `self` to `T`.
//
// FIXME: Clarify interaction with SourceIter/InPlaceIterable. Calling `SouceIter::as_inner`
// after `__iterator_get_unchecked` is supposed to be allowed.
#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
pub unsafe trait TrustedRandomAccess: TrustedRandomAccessNoCoerce {}

/// Like [`TrustedRandomAccess`] but without any of the requirements / guarantees around
/// coercions to subtypes after `__iterator_get_unchecked` (they aren’t allowed here!), and
/// without the requirement that subtypes / supertypes implement `TrustedRandomAccessNoCoerce`.
///
/// This trait was created in PR #85874 to fix soundness issue #85873 without performance regressions.
/// It is subject to change as we might want to build a more generally useful (for performance
/// optimizations) and more sophisticated trait or trait hierarchy that replaces or extends
/// [`TrustedRandomAccess`] and `TrustedRandomAccessNoCoerce`.
#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
#[rustc_specialization_trait]
pub unsafe trait TrustedRandomAccessNoCoerce: Sized {
    // Convenience method.
    fn size(&self) -> usize
    where
        Self: Iterator,
    {
        self.size_hint().0
    }
    /// `true` if getting an iterator element may have side effects.
    /// Remember to take inner iterators into account.
    const MAY_HAVE_SIDE_EFFECT: bool;
}

/// Like `Iterator::__iterator_get_unchecked`, but doesn't require the compiler to
/// know that `U: TrustedRandomAccess`.
///
/// ## Safety
///
/// Same requirements calling `get_unchecked` directly.
#[doc(hidden)]
pub(in crate::iter::adapters) unsafe fn try_get_unchecked<I>(it: &mut I, idx: usize) -> I::Item
where
    I: Iterator,
{
    // SAFETY: the caller must uphold the contract for
    // `Iterator::__iterator_get_unchecked`.
    unsafe { it.try_get_unchecked(idx) }
}

unsafe trait SpecTrustedRandomAccess: Iterator {
    /// If `Self: TrustedRandomAccess`, it must be safe to call
    /// `Iterator::__iterator_get_unchecked(self, index)`.
    unsafe fn try_get_unchecked(&mut self, index: usize) -> Self::Item;
}

unsafe impl<I: Iterator> SpecTrustedRandomAccess for I {
    default unsafe fn try_get_unchecked(&mut self, _: usize) -> Self::Item {
        panic!("Should only be called on TrustedRandomAccess iterators");
    }
}

unsafe impl<I: Iterator + TrustedRandomAccessNoCoerce> SpecTrustedRandomAccess for I {
    unsafe fn try_get_unchecked(&mut self, index: usize) -> Self::Item {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { self.__iterator_get_unchecked(index) }
    }
}
