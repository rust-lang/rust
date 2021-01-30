use crate::intrinsics;
use crate::iter::adapters::{zip::try_get_unchecked, SourceIter, TrustedRandomAccess};
use crate::iter::{FusedIterator, InPlaceIterable, TrustedLen};
use crate::ops::{Add, AddAssign, Try};

/// An iterator that yields the current count and the element during iteration.
///
/// This `struct` is created by the [`enumerate`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`enumerate`]: Iterator::enumerate
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Enumerate<I> {
    iter: I,
    count: usize,
    len: usize,
}
impl<I: Iterator> Enumerate<I> {
    pub(in crate::iter) fn new(iter: I) -> Enumerate<I> {
        EnumerateImpl::new(iter)
    }
}

/// Enumerate specialization trait
///
/// This exists to work around https://bugs.llvm.org/show_bug.cgi?id=48965. It can be removed again
/// once this is solved in LLVM and the implementation of the trait functions can be folded again
/// into the corresponding functions on `Enumerate` based on the default implementation.
///
/// The trait is implemented via specialization on any iterator that implements `TrustedRandomAccess`
/// to provide the information about the maximum value this iterator can return to the optimizer.
/// Specifically, for slices this allows the optimizer to know that the returned values are never
/// bigger than the size of the slice.
///
/// The only difference between the default and specialized implementation is the use of
/// `intrinsics::assume()` on the to be returned values, and both implementations must be kept in
/// sync.
#[doc(hidden)]
trait EnumerateImpl<I> {
    type Item;
    fn new(iter: I) -> Self;
    fn next(&mut self) -> Option<Self::Item>;
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess;
    fn next_back(&mut self) -> Option<Self::Item>
    where
        I: ExactSizeIterator + DoubleEndedIterator;
}

impl<I> EnumerateImpl<I> for Enumerate<I>
where
    I: Iterator,
{
    type Item = (usize, I::Item);

    default fn new(iter: I) -> Self {
        Enumerate {
            iter,
            count: 0,
            len: 0, // unused
        }
    }

    #[inline]
    default fn next(&mut self) -> Option<Self::Item> {
        let a = self.iter.next()?;
        let i = self.count;
        // Possible undefined overflow. By directly calling the trait method instead of using the
        // `+=` operator the decision about overflow checking is delayed to the crate that does code
        // generation, even if overflow checks are disabled for the current crate. This is
        // especially useful because overflow checks are usually disabled for the standard library.
        AddAssign::add_assign(&mut self.count, 1);
        Some((i, a))
    }

    #[inline]
    default unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        let value = unsafe { try_get_unchecked(&mut self.iter, idx) };
        // See comment in `next()` for the reason why `Add::add()` is used here instead of `+`.
        (Add::add(self.count, idx), value)
    }

    #[inline]
    default fn next_back(&mut self) -> Option<Self::Item>
    where
        I: ExactSizeIterator + DoubleEndedIterator,
    {
        let a = self.iter.next_back()?;
        let len = self.iter.len();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len, a))
    }
}

// This is the same code as above but using `intrinsics::assume()` to hint at the compiler
// that the returned index is smaller than the length of the underlying iterator.
//
// This could be bound to `TrustedLen + ExactSizeIterator` or `TrustedRandomAccess` to guarantee
// that the number of elements fits into an `usize` and that the returned length is actually the
// real length. `TrustedRandomAccess` was selected because specialization on `ExactSizeIterator` is
// not possible (yet?).
impl<I> EnumerateImpl<I> for Enumerate<I>
where
    I: TrustedRandomAccess + Iterator,
{
    fn new(iter: I) -> Self {
        let len = iter.size();

        Enumerate { iter, count: 0, len }
    }

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let a = self.iter.next()?;
        // SAFETY: There must be fewer than `self.len` items because of `TrustedLen`'s API contract
        unsafe {
            intrinsics::assume(self.count < self.len);
        }
        let i = self.count;
        // See comment in `next()` of the default implementation for the reason why
        // `AddAssign::add_assign()` is used here instead of `+=`.
        AddAssign::add_assign(&mut self.count, 1);
        Some((i, a))
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        let value = unsafe { try_get_unchecked(&mut self.iter, idx) };
        // See comment in `next()` for the reason why `Add::add()` is used here instead of `+`.
        let idx = Add::add(self.count, idx);
        // SAFETY: There must be fewer than `self.len` items because of `TrustedLen`'s API contract
        unsafe {
            intrinsics::assume(idx < self.len);
        }
        (idx, value)
    }

    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>
    where
        I: ExactSizeIterator + DoubleEndedIterator,
    {
        let a = self.iter.next_back()?;
        let len = self.iter.len();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        let idx = self.count + len;
        // SAFETY: There must be fewer than `self.len` items because of `TrustedLen`'s API contract
        unsafe {
            intrinsics::assume(idx < self.len);
        }
        Some((idx, a))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Enumerate<I>
where
    I: Iterator,
{
    type Item = (usize, <I as Iterator>::Item);

    /// # Overflow Behavior
    ///
    /// The method does no guarding against overflows, so enumerating more than
    /// `usize::MAX` elements either produces the wrong result or panics. If
    /// debug assertions are enabled, a panic is guaranteed.
    ///
    /// # Panics
    ///
    /// Might panic if the index of the element overflows a `usize`.
    #[inline]
    fn next(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        EnumerateImpl::next(self)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<(usize, I::Item)> {
        let a = self.iter.nth(n)?;
        // Possible undefined overflow.
        let i = Add::add(self.count, n);
        self.count = Add::add(i, 1);
        Some((i, a))
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn enumerate<'a, T, Acc, R>(
            count: &'a mut usize,
            mut fold: impl FnMut(Acc, (usize, T)) -> R + 'a,
        ) -> impl FnMut(Acc, T) -> R + 'a {
            move |acc, item| {
                let acc = fold(acc, (*count, item));
                // Possible undefined overflow.
                AddAssign::add_assign(count, 1);
                acc
            }
        }

        self.iter.try_fold(init, enumerate(&mut self.count, fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn enumerate<T, Acc>(
            mut count: usize,
            mut fold: impl FnMut(Acc, (usize, T)) -> Acc,
        ) -> impl FnMut(Acc, T) -> Acc {
            move |acc, item| {
                let acc = fold(acc, (count, item));
                // Possible undefined overflow.
                AddAssign::add_assign(&mut count, 1);
                acc
            }
        }

        self.iter.fold(init, enumerate(self.count, fold))
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> <Self as Iterator>::Item
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: Just forwarding to the actual implementation.
        unsafe { EnumerateImpl::__iterator_get_unchecked(self, idx) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Enumerate<I>
where
    I: ExactSizeIterator + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        EnumerateImpl::next_back(self)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<(usize, <I as Iterator>::Item)> {
        let a = self.iter.nth_back(n)?;
        let len = self.iter.len();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len, a))
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        // Can safely add and subtract the count, as `ExactSizeIterator` promises
        // that the number of elements fits into a `usize`.
        fn enumerate<T, Acc, R>(
            mut count: usize,
            mut fold: impl FnMut(Acc, (usize, T)) -> R,
        ) -> impl FnMut(Acc, T) -> R {
            move |acc, item| {
                count -= 1;
                fold(acc, (count, item))
            }
        }

        let count = self.count + self.iter.len();
        self.iter.try_rfold(init, enumerate(count, fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        // Can safely add and subtract the count, as `ExactSizeIterator` promises
        // that the number of elements fits into a `usize`.
        fn enumerate<T, Acc>(
            mut count: usize,
            mut fold: impl FnMut(Acc, (usize, T)) -> Acc,
        ) -> impl FnMut(Acc, T) -> Acc {
            move |acc, item| {
                count -= 1;
                fold(acc, (count, item))
            }
        }

        let count = self.count + self.iter.len();
        self.iter.rfold(init, enumerate(count, fold))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Enumerate<I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccess for Enumerate<I>
where
    I: TrustedRandomAccess,
{
    fn may_have_side_effect() -> bool {
        I::may_have_side_effect()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Enumerate<I> where I: FusedIterator {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Enumerate<I> where I: TrustedLen {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: Iterator> SourceIter for Enumerate<I>
where
    I: SourceIter<Source = S>,
{
    type Source = S;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut S {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable> InPlaceIterable for Enumerate<I> {}
