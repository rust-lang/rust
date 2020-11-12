use crate::cmp;
use crate::fmt;
use crate::intrinsics;
use crate::ops::{Add, AddAssign, ControlFlow, Try};

use super::from_fn;
use super::{
    DoubleEndedIterator, ExactSizeIterator, FusedIterator, InPlaceIterable, Iterator, TrustedLen,
};

mod chain;
mod flatten;
mod fuse;
mod zip;

pub use self::chain::Chain;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::flatten::{FlatMap, Flatten};
pub use self::fuse::Fuse;
use self::zip::try_get_unchecked;
#[unstable(feature = "trusted_random_access", issue = "none")]
pub use self::zip::TrustedRandomAccess;
pub use self::zip::Zip;

/// This trait provides transitive access to source-stage in an interator-adapter pipeline
/// under the conditions that
/// * the iterator source `S` itself implements `SourceIter<Source = S>`
/// * there is a delegating implementation of this trait for each adapter in the pipeline between
///   the source and the pipeline consumer.
///
/// When the source is an owning iterator struct (commonly called `IntoIter`) then
/// this can be useful for specializing [`FromIterator`] implementations or recovering the
/// remaining elements after an iterator has been partially exhausted.
///
/// Note that implementations do not necessarily have to provide access to the inner-most
/// source of a pipeline. A stateful intermediate adapter might eagerly evaluate a part
/// of the pipeline and expose its internal storage as source.
///
/// The trait is unsafe because implementers must uphold additional safety properties.
/// See [`as_inner`] for details.
///
/// # Examples
///
/// Retrieving a partially consumed source:
///
/// ```
/// # #![feature(inplace_iteration)]
/// # use std::iter::SourceIter;
///
/// let mut iter = vec![9, 9, 9].into_iter().map(|i| i * i);
/// let _ = iter.next();
/// let mut remainder = std::mem::replace(unsafe { iter.as_inner() }, Vec::new().into_iter());
/// println!("n = {} elements remaining", remainder.len());
/// ```
///
/// [`FromIterator`]: crate::iter::FromIterator
/// [`as_inner`]: SourceIter::as_inner
#[unstable(issue = "none", feature = "inplace_iteration")]
pub unsafe trait SourceIter {
    /// A source stage in an iterator pipeline.
    type Source: Iterator;

    /// Retrieve the source of an iterator pipeline.
    ///
    /// # Safety
    ///
    /// Implementations of must return the same mutable reference for their lifetime, unless
    /// replaced by a caller.
    /// Callers may only replace the reference when they stopped iteration and drop the
    /// iterator pipeline after extracting the source.
    ///
    /// This means iterator adapters can rely on the source not changing during
    /// iteration but they cannot rely on it in their Drop implementations.
    ///
    /// Implementing this method means adapters relinquish private-only access to their
    /// source and can only rely on guarantees made based on method receiver types.
    /// The lack of restricted access also requires that adapters must uphold the source's
    /// public API even when they have access to its internals.
    ///
    /// Callers in turn must expect the source to be in any state that is consistent with
    /// its public API since adapters sitting between it and the source have the same
    /// access. In particular an adapter may have consumed more elements than strictly necessary.
    ///
    /// The overall goal of these requirements is to let the consumer of a pipeline use
    /// * whatever remains in the source after iteration has stopped
    /// * the memory that has become unused by advancing a consuming iterator
    ///
    /// [`next()`]: Iterator::next
    unsafe fn as_inner(&mut self) -> &mut Self::Source;
}

/// A double-ended iterator with the direction inverted.
///
/// This `struct` is created by the [`rev`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`rev`]: Iterator::rev
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Rev<T> {
    iter: T,
}
impl<T> Rev<T> {
    pub(super) fn new(iter: T) -> Rev<T> {
        Rev { iter }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Rev<I>
where
    I: DoubleEndedIterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        self.iter.next_back()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        self.iter.advance_back_by(n)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        self.iter.nth_back(n)
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.iter.try_rfold(init, f)
    }

    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, f)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.iter.rfind(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Rev<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        self.iter.next()
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
        self.iter.advance_by(n)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        self.iter.nth(n)
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.iter.try_fold(init, f)
    }

    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, f)
    }

    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.iter.find(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Rev<I>
where
    I: ExactSizeIterator + DoubleEndedIterator,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Rev<I> where I: FusedIterator + DoubleEndedIterator {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Rev<I> where I: TrustedLen + DoubleEndedIterator {}

/// An iterator that copies the elements of an underlying iterator.
///
/// This `struct` is created by the [`copied`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`copied`]: Iterator::copied
/// [`Iterator`]: trait.Iterator.html
#[stable(feature = "iter_copied", since = "1.36.0")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct Copied<I> {
    it: I,
}

impl<I> Copied<I> {
    pub(super) fn new(it: I) -> Copied<I> {
        Copied { it }
    }
}

fn copy_fold<T: Copy, Acc>(mut f: impl FnMut(Acc, T) -> Acc) -> impl FnMut(Acc, &T) -> Acc {
    move |acc, &elt| f(acc, elt)
}

fn copy_try_fold<T: Copy, Acc, R>(mut f: impl FnMut(Acc, T) -> R) -> impl FnMut(Acc, &T) -> R {
    move |acc, &elt| f(acc, elt)
}

#[stable(feature = "iter_copied", since = "1.36.0")]
impl<'a, I, T: 'a> Iterator for Copied<I>
where
    I: Iterator<Item = &'a T>,
    T: Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.it.next().copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.it.try_fold(init, copy_try_fold(f))
    }

    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.fold(init, copy_fold(f))
    }

    fn nth(&mut self, n: usize) -> Option<T> {
        self.it.nth(n).copied()
    }

    fn last(self) -> Option<T> {
        self.it.last().copied()
    }

    fn count(self) -> usize {
        self.it.count()
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> T
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        *unsafe { try_get_unchecked(&mut self.it, idx) }
    }
}

#[stable(feature = "iter_copied", since = "1.36.0")]
impl<'a, I, T: 'a> DoubleEndedIterator for Copied<I>
where
    I: DoubleEndedIterator<Item = &'a T>,
    T: Copy,
{
    fn next_back(&mut self) -> Option<T> {
        self.it.next_back().copied()
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.it.try_rfold(init, copy_try_fold(f))
    }

    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.rfold(init, copy_fold(f))
    }
}

#[stable(feature = "iter_copied", since = "1.36.0")]
impl<'a, I, T: 'a> ExactSizeIterator for Copied<I>
where
    I: ExactSizeIterator<Item = &'a T>,
    T: Copy,
{
    fn len(&self) -> usize {
        self.it.len()
    }

    fn is_empty(&self) -> bool {
        self.it.is_empty()
    }
}

#[stable(feature = "iter_copied", since = "1.36.0")]
impl<'a, I, T: 'a> FusedIterator for Copied<I>
where
    I: FusedIterator<Item = &'a T>,
    T: Copy,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccess for Copied<I>
where
    I: TrustedRandomAccess,
{
    #[inline]
    fn may_have_side_effect() -> bool {
        I::may_have_side_effect()
    }
}

#[stable(feature = "iter_copied", since = "1.36.0")]
unsafe impl<'a, I, T: 'a> TrustedLen for Copied<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Copy,
{
}

/// An iterator that clones the elements of an underlying iterator.
///
/// This `struct` is created by the [`cloned`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`cloned`]: Iterator::cloned
/// [`Iterator`]: trait.Iterator.html
#[stable(feature = "iter_cloned", since = "1.1.0")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct Cloned<I> {
    it: I,
}
impl<I> Cloned<I> {
    pub(super) fn new(it: I) -> Cloned<I> {
        Cloned { it }
    }
}

fn clone_try_fold<T: Clone, Acc, R>(mut f: impl FnMut(Acc, T) -> R) -> impl FnMut(Acc, &T) -> R {
    move |acc, elt| f(acc, elt.clone())
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> Iterator for Cloned<I>
where
    I: Iterator<Item = &'a T>,
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.it.next().cloned()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.it.try_fold(init, clone_try_fold(f))
    }

    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.map(T::clone).fold(init, f)
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> T
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { try_get_unchecked(&mut self.it, idx).clone() }
    }
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> DoubleEndedIterator for Cloned<I>
where
    I: DoubleEndedIterator<Item = &'a T>,
    T: Clone,
{
    fn next_back(&mut self) -> Option<T> {
        self.it.next_back().cloned()
    }

    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.it.try_rfold(init, clone_try_fold(f))
    }

    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.it.map(T::clone).rfold(init, f)
    }
}

#[stable(feature = "iter_cloned", since = "1.1.0")]
impl<'a, I, T: 'a> ExactSizeIterator for Cloned<I>
where
    I: ExactSizeIterator<Item = &'a T>,
    T: Clone,
{
    fn len(&self) -> usize {
        self.it.len()
    }

    fn is_empty(&self) -> bool {
        self.it.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<'a, I, T: 'a> FusedIterator for Cloned<I>
where
    I: FusedIterator<Item = &'a T>,
    T: Clone,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I> TrustedRandomAccess for Cloned<I>
where
    I: TrustedRandomAccess,
{
    #[inline]
    fn may_have_side_effect() -> bool {
        true
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<'a, I, T: 'a> TrustedLen for Cloned<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Clone,
{
}

/// An iterator that repeats endlessly.
///
/// This `struct` is created by the [`cycle`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`cycle`]: Iterator::cycle
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Cycle<I> {
    orig: I,
    iter: I,
}
impl<I: Clone> Cycle<I> {
    pub(super) fn new(iter: I) -> Cycle<I> {
        Cycle { orig: iter.clone(), iter }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Cycle<I>
where
    I: Clone + Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        match self.iter.next() {
            None => {
                self.iter = self.orig.clone();
                self.iter.next()
            }
            y => y,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // the cycle iterator is either empty or infinite
        match self.orig.size_hint() {
            sz @ (0, Some(0)) => sz,
            (0, _) => (0, None),
            _ => (usize::MAX, None),
        }
    }

    #[inline]
    fn try_fold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        // fully iterate the current iterator. this is necessary because
        // `self.iter` may be empty even when `self.orig` isn't
        acc = self.iter.try_fold(acc, &mut f)?;
        self.iter = self.orig.clone();

        // complete a full cycle, keeping track of whether the cycled
        // iterator is empty or not. we need to return early in case
        // of an empty iterator to prevent an infinite loop
        let mut is_empty = true;
        acc = self.iter.try_fold(acc, |acc, x| {
            is_empty = false;
            f(acc, x)
        })?;

        if is_empty {
            return try { acc };
        }

        loop {
            self.iter = self.orig.clone();
            acc = self.iter.try_fold(acc, &mut f)?;
        }
    }

    // No `fold` override, because `fold` doesn't make much sense for `Cycle`,
    // and we can't do anything better than the default.
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Cycle<I> where I: Clone + Iterator {}

/// An iterator for stepping iterators by a custom amount.
///
/// This `struct` is created by the [`step_by`] method on [`Iterator`]. See
/// its documentation for more.
///
/// [`step_by`]: Iterator::step_by
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iterator_step_by", since = "1.28.0")]
#[derive(Clone, Debug)]
pub struct StepBy<I> {
    iter: I,
    step: usize,
    first_take: bool,
}
impl<I> StepBy<I> {
    pub(super) fn new(iter: I, step: usize) -> StepBy<I> {
        assert!(step != 0);
        StepBy { iter, step: step - 1, first_take: true }
    }
}

#[stable(feature = "iterator_step_by", since = "1.28.0")]
impl<I> Iterator for StepBy<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            self.iter.next()
        } else {
            self.iter.nth(self.step)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        #[inline]
        fn first_size(step: usize) -> impl Fn(usize) -> usize {
            move |n| if n == 0 { 0 } else { 1 + (n - 1) / (step + 1) }
        }

        #[inline]
        fn other_size(step: usize) -> impl Fn(usize) -> usize {
            move |n| n / (step + 1)
        }

        let (low, high) = self.iter.size_hint();

        if self.first_take {
            let f = first_size(self.step);
            (f(low), high.map(f))
        } else {
            let f = other_size(self.step);
            (f(low), high.map(f))
        }
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            let first = self.iter.next();
            if n == 0 {
                return first;
            }
            n -= 1;
        }
        // n and self.step are indices, we need to add 1 to get the amount of elements
        // When calling `.nth`, we need to subtract 1 again to convert back to an index
        // step + 1 can't overflow because `.step_by` sets `self.step` to `step - 1`
        let mut step = self.step + 1;
        // n + 1 could overflow
        // thus, if n is usize::MAX, instead of adding one, we call .nth(step)
        if n == usize::MAX {
            self.iter.nth(step - 1);
        } else {
            n += 1;
        }

        // overflow handling
        loop {
            let mul = n.checked_mul(step);
            {
                if intrinsics::likely(mul.is_some()) {
                    return self.iter.nth(mul.unwrap() - 1);
                }
            }
            let div_n = usize::MAX / n;
            let div_step = usize::MAX / step;
            let nth_n = div_n * n;
            let nth_step = div_step * step;
            let nth = if nth_n > nth_step {
                step -= div_n;
                nth_n
            } else {
                n -= div_step;
                nth_step
            };
            self.iter.nth(nth - 1);
        }
    }

    fn try_fold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn nth<I: Iterator>(iter: &mut I, step: usize) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth(step)
        }

        if self.first_take {
            self.first_take = false;
            match self.iter.next() {
                None => return try { acc },
                Some(x) => acc = f(acc, x)?,
            }
        }
        from_fn(nth(&mut self.iter, self.step)).try_fold(acc, f)
    }

    fn fold<Acc, F>(mut self, mut acc: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn nth<I: Iterator>(iter: &mut I, step: usize) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth(step)
        }

        if self.first_take {
            self.first_take = false;
            match self.iter.next() {
                None => return acc,
                Some(x) => acc = f(acc, x),
            }
        }
        from_fn(nth(&mut self.iter, self.step)).fold(acc, f)
    }
}

impl<I> StepBy<I>
where
    I: ExactSizeIterator,
{
    // The zero-based index starting from the end of the iterator of the
    // last element. Used in the `DoubleEndedIterator` implementation.
    fn next_back_index(&self) -> usize {
        let rem = self.iter.len() % (self.step + 1);
        if self.first_take {
            if rem == 0 { self.step } else { rem - 1 }
        } else {
            rem
        }
    }
}

#[stable(feature = "double_ended_step_by_iterator", since = "1.38.0")]
impl<I> DoubleEndedIterator for StepBy<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.nth_back(self.next_back_index())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // `self.iter.nth_back(usize::MAX)` does the right thing here when `n`
        // is out of bounds because the length of `self.iter` does not exceed
        // `usize::MAX` (because `I: ExactSizeIterator`) and `nth_back` is
        // zero-indexed
        let n = n.saturating_mul(self.step + 1).saturating_add(self.next_back_index());
        self.iter.nth_back(n)
    }

    fn try_rfold<Acc, F, R>(&mut self, init: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn nth_back<I: DoubleEndedIterator>(
            iter: &mut I,
            step: usize,
        ) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth_back(step)
        }

        match self.next_back() {
            None => try { init },
            Some(x) => {
                let acc = f(init, x)?;
                from_fn(nth_back(&mut self.iter, self.step)).try_fold(acc, f)
            }
        }
    }

    #[inline]
    fn rfold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn nth_back<I: DoubleEndedIterator>(
            iter: &mut I,
            step: usize,
        ) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth_back(step)
        }

        match self.next_back() {
            None => init,
            Some(x) => {
                let acc = f(init, x);
                from_fn(nth_back(&mut self.iter, self.step)).fold(acc, f)
            }
        }
    }
}

// StepBy can only make the iterator shorter, so the len will still fit.
#[stable(feature = "iterator_step_by", since = "1.28.0")]
impl<I> ExactSizeIterator for StepBy<I> where I: ExactSizeIterator {}

/// An iterator that maps the values of `iter` with `f`.
///
/// This `struct` is created by the [`map`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`map`]: Iterator::map
/// [`Iterator`]: trait.Iterator.html
///
/// # Notes about side effects
///
/// The [`map`] iterator implements [`DoubleEndedIterator`], meaning that
/// you can also [`map`] backwards:
///
/// ```rust
/// let v: Vec<i32> = vec![1, 2, 3].into_iter().map(|x| x + 1).rev().collect();
///
/// assert_eq!(v, [4, 3, 2]);
/// ```
///
/// [`DoubleEndedIterator`]: trait.DoubleEndedIterator.html
///
/// But if your closure has state, iterating backwards may act in a way you do
/// not expect. Let's go through an example. First, in the forward direction:
///
/// ```rust
/// let mut c = 0;
///
/// for pair in vec!['a', 'b', 'c'].into_iter()
///                                .map(|letter| { c += 1; (letter, c) }) {
///     println!("{:?}", pair);
/// }
/// ```
///
/// This will print "('a', 1), ('b', 2), ('c', 3)".
///
/// Now consider this twist where we add a call to `rev`. This version will
/// print `('c', 1), ('b', 2), ('a', 3)`. Note that the letters are reversed,
/// but the values of the counter still go in order. This is because `map()` is
/// still being called lazily on each item, but we are popping items off the
/// back of the vector now, instead of shifting them from the front.
///
/// ```rust
/// let mut c = 0;
///
/// for pair in vec!['a', 'b', 'c'].into_iter()
///                                .map(|letter| { c += 1; (letter, c) })
///                                .rev() {
///     println!("{:?}", pair);
/// }
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Map<I, F> {
    iter: I,
    f: F,
}
impl<I, F> Map<I, F> {
    pub(super) fn new(iter: I, f: F) -> Map<I, F> {
        Map { iter, f }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, F> fmt::Debug for Map<I, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Map").field("iter", &self.iter).finish()
    }
}

fn map_fold<T, B, Acc>(
    mut f: impl FnMut(T) -> B,
    mut g: impl FnMut(Acc, B) -> Acc,
) -> impl FnMut(Acc, T) -> Acc {
    move |acc, elt| g(acc, f(elt))
}

fn map_try_fold<'a, T, B, Acc, R>(
    f: &'a mut impl FnMut(T) -> B,
    mut g: impl FnMut(Acc, B) -> R + 'a,
) -> impl FnMut(Acc, T) -> R + 'a {
    move |acc, elt| g(acc, f(elt))
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: Iterator, F> Iterator for Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().map(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn try_fold<Acc, G, R>(&mut self, init: Acc, g: G) -> R
    where
        Self: Sized,
        G: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_fold(init, map_try_fold(&mut self.f, g))
    }

    fn fold<Acc, G>(self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, map_fold(self.f, g))
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> B
    where
        Self: TrustedRandomAccess,
    {
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        unsafe { (self.f)(try_get_unchecked(&mut self.iter, idx)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: DoubleEndedIterator, F> DoubleEndedIterator for Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        self.iter.next_back().map(&mut self.f)
    }

    fn try_rfold<Acc, G, R>(&mut self, init: Acc, g: G) -> R
    where
        Self: Sized,
        G: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_rfold(init, map_try_fold(&mut self.f, g))
    }

    fn rfold<Acc, G>(self, init: Acc, g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, map_fold(self.f, g))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: ExactSizeIterator, F> ExactSizeIterator for Map<I, F>
where
    F: FnMut(I::Item) -> B,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<B, I: FusedIterator, F> FusedIterator for Map<I, F> where F: FnMut(I::Item) -> B {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<B, I, F> TrustedLen for Map<I, F>
where
    I: TrustedLen,
    F: FnMut(I::Item) -> B,
{
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<I, F> TrustedRandomAccess for Map<I, F>
where
    I: TrustedRandomAccess,
{
    #[inline]
    fn may_have_side_effect() -> bool {
        true
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, B, I: Iterator, F> SourceIter for Map<I, F>
where
    F: FnMut(I::Item) -> B,
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
unsafe impl<B, I: InPlaceIterable, F> InPlaceIterable for Map<I, F> where F: FnMut(I::Item) -> B {}

/// An iterator that filters the elements of `iter` with `predicate`.
///
/// This `struct` is created by the [`filter`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`filter`]: Iterator::filter
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Filter<I, P> {
    iter: I,
    predicate: P,
}
impl<I, P> Filter<I, P> {
    pub(super) fn new(iter: I, predicate: P) -> Filter<I, P> {
        Filter { iter, predicate }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, P> fmt::Debug for Filter<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Filter").field("iter", &self.iter).finish()
    }
}

fn filter_fold<T, Acc>(
    mut predicate: impl FnMut(&T) -> bool,
    mut fold: impl FnMut(Acc, T) -> Acc,
) -> impl FnMut(Acc, T) -> Acc {
    move |acc, item| if predicate(&item) { fold(acc, item) } else { acc }
}

fn filter_try_fold<'a, T, Acc, R: Try<Ok = Acc>>(
    predicate: &'a mut impl FnMut(&T) -> bool,
    mut fold: impl FnMut(Acc, T) -> R + 'a,
) -> impl FnMut(Acc, T) -> R + 'a {
    move |acc, item| if predicate(&item) { fold(acc, item) } else { try { acc } }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for Filter<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        self.iter.find(&mut self.predicate)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }

    // this special case allows the compiler to make `.filter(_).count()`
    // branchless. Barring perfect branch prediction (which is unattainable in
    // the general case), this will be much faster in >90% of cases (containing
    // virtually all real workloads) and only a tiny bit slower in the rest.
    //
    // Having this specialization thus allows us to write `.filter(p).count()`
    // where we would otherwise write `.map(|x| p(x) as usize).sum()`, which is
    // less readable and also less backwards-compatible to Rust before 1.10.
    //
    // Using the branchless version will also simplify the LLVM byte code, thus
    // leaving more budget for LLVM optimizations.
    #[inline]
    fn count(self) -> usize {
        #[inline]
        fn to_usize<T>(mut predicate: impl FnMut(&T) -> bool) -> impl FnMut(T) -> usize {
            move |x| predicate(&x) as usize
        }

        self.iter.map(to_usize(self.predicate)).sum()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_fold(init, filter_try_fold(&mut self.predicate, fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, filter_fold(self.predicate, fold))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, P> DoubleEndedIterator for Filter<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        self.iter.rfind(&mut self.predicate)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_rfold(init, filter_try_fold(&mut self.predicate, fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, filter_fold(self.predicate, fold))
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator, P> FusedIterator for Filter<I, P> where P: FnMut(&I::Item) -> bool {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, P, I: Iterator> SourceIter for Filter<I, P>
where
    P: FnMut(&I::Item) -> bool,
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
unsafe impl<I: InPlaceIterable, P> InPlaceIterable for Filter<I, P> where P: FnMut(&I::Item) -> bool {}

/// An iterator that uses `f` to both filter and map elements from `iter`.
///
/// This `struct` is created by the [`filter_map`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`filter_map`]: Iterator::filter_map
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct FilterMap<I, F> {
    iter: I,
    f: F,
}
impl<I, F> FilterMap<I, F> {
    pub(super) fn new(iter: I, f: F) -> FilterMap<I, F> {
        FilterMap { iter, f }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, F> fmt::Debug for FilterMap<I, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterMap").field("iter", &self.iter).finish()
    }
}

fn filter_map_fold<T, B, Acc>(
    mut f: impl FnMut(T) -> Option<B>,
    mut fold: impl FnMut(Acc, B) -> Acc,
) -> impl FnMut(Acc, T) -> Acc {
    move |acc, item| match f(item) {
        Some(x) => fold(acc, x),
        None => acc,
    }
}

fn filter_map_try_fold<'a, T, B, Acc, R: Try<Ok = Acc>>(
    f: &'a mut impl FnMut(T) -> Option<B>,
    mut fold: impl FnMut(Acc, B) -> R + 'a,
) -> impl FnMut(Acc, T) -> R + 'a {
    move |acc, item| match f(item) {
        Some(x) => fold(acc, x),
        None => try { acc },
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: Iterator, F> Iterator for FilterMap<I, F>
where
    F: FnMut(I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.find_map(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_fold(init, filter_map_try_fold(&mut self.f, fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, filter_map_fold(self.f, fold))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I: DoubleEndedIterator, F> DoubleEndedIterator for FilterMap<I, F>
where
    F: FnMut(I::Item) -> Option<B>,
{
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        #[inline]
        fn find<T, B>(
            f: &mut impl FnMut(T) -> Option<B>,
        ) -> impl FnMut((), T) -> ControlFlow<B> + '_ {
            move |(), x| match f(x) {
                Some(x) => ControlFlow::Break(x),
                None => ControlFlow::CONTINUE,
            }
        }

        self.iter.try_rfold((), find(&mut self.f)).break_value()
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_rfold(init, filter_map_try_fold(&mut self.f, fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, filter_map_fold(self.f, fold))
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<B, I: FusedIterator, F> FusedIterator for FilterMap<I, F> where F: FnMut(I::Item) -> Option<B> {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, B, I: Iterator, F> SourceIter for FilterMap<I, F>
where
    F: FnMut(I::Item) -> Option<B>,
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
unsafe impl<B, I: InPlaceIterable, F> InPlaceIterable for FilterMap<I, F> where
    F: FnMut(I::Item) -> Option<B>
{
}

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
}
impl<I> Enumerate<I> {
    pub(super) fn new(iter: I) -> Enumerate<I> {
        Enumerate { iter, count: 0 }
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
        let a = self.iter.next()?;
        let i = self.count;
        // Possible undefined overflow.
        AddAssign::add_assign(&mut self.count, 1);
        Some((i, a))
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
        // SAFETY: the caller must uphold the contract for
        // `Iterator::__iterator_get_unchecked`.
        let value = unsafe { try_get_unchecked(&mut self.iter, idx) };
        (Add::add(self.count, idx), value)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Enumerate<I>
where
    I: ExactSizeIterator + DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        let a = self.iter.next_back()?;
        let len = self.iter.len();
        // Can safely add, `ExactSizeIterator` promises that the number of
        // elements fits into a `usize`.
        Some((self.count + len, a))
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

/// An iterator with a `peek()` that returns an optional reference to the next
/// element.
///
/// This `struct` is created by the [`peekable`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`peekable`]: Iterator::peekable
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Peekable<I: Iterator> {
    iter: I,
    /// Remember a peeked value, even if it was None.
    peeked: Option<Option<I::Item>>,
}
impl<I: Iterator> Peekable<I> {
    pub(super) fn new(iter: I) -> Peekable<I> {
        Peekable { iter, peeked: None }
    }
}

// Peekable must remember if a None has been seen in the `.peek()` method.
// It ensures that `.peek(); .peek();` or `.peek(); .next();` only advances the
// underlying iterator at most once. This does not by itself make the iterator
// fused.
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        match self.peeked.take() {
            Some(v) => v,
            None => self.iter.next(),
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn count(mut self) -> usize {
        match self.peeked.take() {
            Some(None) => 0,
            Some(Some(_)) => 1 + self.iter.count(),
            None => self.iter.count(),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        match self.peeked.take() {
            Some(None) => None,
            Some(v @ Some(_)) if n == 0 => v,
            Some(Some(_)) => self.iter.nth(n - 1),
            None => self.iter.nth(n),
        }
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        let peek_opt = match self.peeked.take() {
            Some(None) => return None,
            Some(v) => v,
            None => None,
        };
        self.iter.last().or(peek_opt)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let peek_len = match self.peeked {
            Some(None) => return (0, Some(0)),
            Some(Some(_)) => 1,
            None => 0,
        };
        let (lo, hi) = self.iter.size_hint();
        let lo = lo.saturating_add(peek_len);
        let hi = match hi {
            Some(x) => x.checked_add(peek_len),
            None => None,
        };
        (lo, hi)
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        let acc = match self.peeked.take() {
            Some(None) => return try { init },
            Some(Some(v)) => f(init, v)?,
            None => init,
        };
        self.iter.try_fold(acc, f)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let acc = match self.peeked {
            Some(None) => return init,
            Some(Some(v)) => fold(init, v),
            None => init,
        };
        self.iter.fold(acc, fold)
    }
}

#[stable(feature = "double_ended_peek_iterator", since = "1.38.0")]
impl<I> DoubleEndedIterator for Peekable<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.peeked.as_mut() {
            Some(v @ Some(_)) => self.iter.next_back().or_else(|| v.take()),
            Some(None) => None,
            None => self.iter.next_back(),
        }
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        match self.peeked.take() {
            Some(None) => try { init },
            Some(Some(v)) => match self.iter.try_rfold(init, &mut f).into_result() {
                Ok(acc) => f(acc, v),
                Err(e) => {
                    self.peeked = Some(Some(v));
                    Try::from_error(e)
                }
            },
            None => self.iter.try_rfold(init, f),
        }
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        match self.peeked {
            Some(None) => init,
            Some(Some(v)) => {
                let acc = self.iter.rfold(init, &mut fold);
                fold(acc, v)
            }
            None => self.iter.rfold(init, fold),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator> ExactSizeIterator for Peekable<I> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator> FusedIterator for Peekable<I> {}

impl<I: Iterator> Peekable<I> {
    /// Returns a reference to the next() value without advancing the iterator.
    ///
    /// Like [`next`], if there is a value, it is wrapped in a `Some(T)`.
    /// But if the iteration is over, `None` is returned.
    ///
    /// [`next`]: Iterator::next
    ///
    /// Because `peek()` returns a reference, and many iterators iterate over
    /// references, there can be a possibly confusing situation where the
    /// return value is a double reference. You can see this effect in the
    /// examples below.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let xs = [1, 2, 3];
    ///
    /// let mut iter = xs.iter().peekable();
    ///
    /// // peek() lets us see into the future
    /// assert_eq!(iter.peek(), Some(&&1));
    /// assert_eq!(iter.next(), Some(&1));
    ///
    /// assert_eq!(iter.next(), Some(&2));
    ///
    /// // The iterator does not advance even if we `peek` multiple times
    /// assert_eq!(iter.peek(), Some(&&3));
    /// assert_eq!(iter.peek(), Some(&&3));
    ///
    /// assert_eq!(iter.next(), Some(&3));
    ///
    /// // After the iterator is finished, so is `peek()`
    /// assert_eq!(iter.peek(), None);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn peek(&mut self) -> Option<&I::Item> {
        let iter = &mut self.iter;
        self.peeked.get_or_insert_with(|| iter.next()).as_ref()
    }

    /// Consume and return the next value of this iterator if a condition is true.
    ///
    /// If `func` returns `true` for the next value of this iterator, consume and return it.
    /// Otherwise, return `None`.
    ///
    /// # Examples
    /// Consume a number if it's equal to 0.
    /// ```
    /// #![feature(peekable_next_if)]
    /// let mut iter = (0..5).peekable();
    /// // The first item of the iterator is 0; consume it.
    /// assert_eq!(iter.next_if(|&x| x == 0), Some(0));
    /// // The next item returned is now 1, so `consume` will return `false`.
    /// assert_eq!(iter.next_if(|&x| x == 0), None);
    /// // `next_if` saves the value of the next item if it was not equal to `expected`.
    /// assert_eq!(iter.next(), Some(1));
    /// ```
    ///
    /// Consume any number less than 10.
    /// ```
    /// #![feature(peekable_next_if)]
    /// let mut iter = (1..20).peekable();
    /// // Consume all numbers less than 10
    /// while iter.next_if(|&x| x < 10).is_some() {}
    /// // The next value returned will be 10
    /// assert_eq!(iter.next(), Some(10));
    /// ```
    #[unstable(feature = "peekable_next_if", issue = "72480")]
    pub fn next_if(&mut self, func: impl FnOnce(&I::Item) -> bool) -> Option<I::Item> {
        match self.next() {
            Some(matched) if func(&matched) => Some(matched),
            other => {
                // Since we called `self.next()`, we consumed `self.peeked`.
                assert!(self.peeked.is_none());
                self.peeked = Some(other);
                None
            }
        }
    }

    /// Consume and return the next item if it is equal to `expected`.
    ///
    /// # Example
    /// Consume a number if it's equal to 0.
    /// ```
    /// #![feature(peekable_next_if)]
    /// let mut iter = (0..5).peekable();
    /// // The first item of the iterator is 0; consume it.
    /// assert_eq!(iter.next_if_eq(&0), Some(0));
    /// // The next item returned is now 1, so `consume` will return `false`.
    /// assert_eq!(iter.next_if_eq(&0), None);
    /// // `next_if_eq` saves the value of the next item if it was not equal to `expected`.
    /// assert_eq!(iter.next(), Some(1));
    /// ```
    #[unstable(feature = "peekable_next_if", issue = "72480")]
    pub fn next_if_eq<T>(&mut self, expected: &T) -> Option<I::Item>
    where
        T: ?Sized,
        I::Item: PartialEq<T>,
    {
        self.next_if(|next| next == expected)
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I> TrustedLen for Peekable<I> where I: TrustedLen {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: Iterator> SourceIter for Peekable<I>
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
unsafe impl<I: InPlaceIterable> InPlaceIterable for Peekable<I> {}

/// An iterator that rejects elements while `predicate` returns `true`.
///
/// This `struct` is created by the [`skip_while`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`skip_while`]: Iterator::skip_while
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct SkipWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}
impl<I, P> SkipWhile<I, P> {
    pub(super) fn new(iter: I, predicate: P) -> SkipWhile<I, P> {
        SkipWhile { iter, flag: false, predicate }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, P> fmt::Debug for SkipWhile<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkipWhile").field("iter", &self.iter).field("flag", &self.flag).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for SkipWhile<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        fn check<'a, T>(
            flag: &'a mut bool,
            pred: &'a mut impl FnMut(&T) -> bool,
        ) -> impl FnMut(&T) -> bool + 'a {
            move |x| {
                if *flag || !pred(x) {
                    *flag = true;
                    true
                } else {
                    false
                }
            }
        }

        let flag = &mut self.flag;
        let pred = &mut self.predicate;
        self.iter.find(check(flag, pred))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, mut init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if !self.flag {
            match self.next() {
                Some(v) => init = fold(init, v)?,
                None => return try { init },
            }
        }
        self.iter.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, mut init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if !self.flag {
            match self.next() {
                Some(v) => init = fold(init, v),
                None => return init,
            }
        }
        self.iter.fold(init, fold)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I, P> FusedIterator for SkipWhile<I, P>
where
    I: FusedIterator,
    P: FnMut(&I::Item) -> bool,
{
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, P, I: Iterator> SourceIter for SkipWhile<I, P>
where
    P: FnMut(&I::Item) -> bool,
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
unsafe impl<I: InPlaceIterable, F> InPlaceIterable for SkipWhile<I, F> where
    F: FnMut(&I::Item) -> bool
{
}

/// An iterator that only accepts elements while `predicate` returns `true`.
///
/// This `struct` is created by the [`take_while`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`take_while`]: Iterator::take_while
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct TakeWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}
impl<I, P> TakeWhile<I, P> {
    pub(super) fn new(iter: I, predicate: P) -> TakeWhile<I, P> {
        TakeWhile { iter, flag: false, predicate }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, P> fmt::Debug for TakeWhile<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TakeWhile").field("iter", &self.iter).field("flag", &self.flag).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, P> Iterator for TakeWhile<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.flag {
            None
        } else {
            let x = self.iter.next()?;
            if (self.predicate)(&x) {
                Some(x)
            } else {
                self.flag = true;
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.flag {
            (0, Some(0))
        } else {
            let (_, upper) = self.iter.size_hint();
            (0, upper) // can't know a lower bound, due to the predicate
        }
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        fn check<'a, T, Acc, R: Try<Ok = Acc>>(
            flag: &'a mut bool,
            p: &'a mut impl FnMut(&T) -> bool,
            mut fold: impl FnMut(Acc, T) -> R + 'a,
        ) -> impl FnMut(Acc, T) -> ControlFlow<R, Acc> + 'a {
            move |acc, x| {
                if p(&x) {
                    ControlFlow::from_try(fold(acc, x))
                } else {
                    *flag = true;
                    ControlFlow::Break(try { acc })
                }
            }
        }

        if self.flag {
            try { init }
        } else {
            let flag = &mut self.flag;
            let p = &mut self.predicate;
            self.iter.try_fold(init, check(flag, p, fold)).into_try()
        }
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(fold)).unwrap()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I, P> FusedIterator for TakeWhile<I, P>
where
    I: FusedIterator,
    P: FnMut(&I::Item) -> bool,
{
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, P, I: Iterator> SourceIter for TakeWhile<I, P>
where
    P: FnMut(&I::Item) -> bool,
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
unsafe impl<I: InPlaceIterable, F> InPlaceIterable for TakeWhile<I, F> where
    F: FnMut(&I::Item) -> bool
{
}

/// An iterator that only accepts elements while `predicate` returns `Some(_)`.
///
/// This `struct` is created by the [`map_while`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`map_while`]: Iterator::map_while
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "iter_map_while", reason = "recently added", issue = "68537")]
#[derive(Clone)]
pub struct MapWhile<I, P> {
    iter: I,
    predicate: P,
}

impl<I, P> MapWhile<I, P> {
    pub(super) fn new(iter: I, predicate: P) -> MapWhile<I, P> {
        MapWhile { iter, predicate }
    }
}

#[unstable(feature = "iter_map_while", reason = "recently added", issue = "68537")]
impl<I: fmt::Debug, P> fmt::Debug for MapWhile<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapWhile").field("iter", &self.iter).finish()
    }
}

#[unstable(feature = "iter_map_while", reason = "recently added", issue = "68537")]
impl<B, I: Iterator, P> Iterator for MapWhile<I, P>
where
    P: FnMut(I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        let x = self.iter.next()?;
        (self.predicate)(x)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        let Self { iter, predicate } = self;
        iter.try_fold(init, |acc, x| match predicate(x) {
            Some(item) => ControlFlow::from_try(fold(acc, item)),
            None => ControlFlow::Break(try { acc }),
        })
        .into_try()
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(fold)).unwrap()
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, B, I: Iterator, P> SourceIter for MapWhile<I, P>
where
    P: FnMut(I::Item) -> Option<B>,
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
unsafe impl<B, I: InPlaceIterable, P> InPlaceIterable for MapWhile<I, P> where
    P: FnMut(I::Item) -> Option<B>
{
}

/// An iterator that skips over `n` elements of `iter`.
///
/// This `struct` is created by the [`skip`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`skip`]: Iterator::skip
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Skip<I> {
    iter: I,
    n: usize,
}
impl<I> Skip<I> {
    pub(super) fn new(iter: I, n: usize) -> Skip<I> {
        Skip { iter, n }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Skip<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.n == 0 {
            self.iter.next()
        } else {
            let old_n = self.n;
            self.n = 0;
            self.iter.nth(old_n)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        // Can't just add n + self.n due to overflow.
        if self.n > 0 {
            let to_skip = self.n;
            self.n = 0;
            // nth(n) skips n+1
            self.iter.nth(to_skip - 1)?;
        }
        self.iter.nth(n)
    }

    #[inline]
    fn count(mut self) -> usize {
        if self.n > 0 {
            // nth(n) skips n+1
            if self.iter.nth(self.n - 1).is_none() {
                return 0;
            }
        }
        self.iter.count()
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        if self.n > 0 {
            // nth(n) skips n+1
            self.iter.nth(self.n - 1)?;
        }
        self.iter.last()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        let lower = lower.saturating_sub(self.n);
        let upper = match upper {
            Some(x) => Some(x.saturating_sub(self.n)),
            None => None,
        };

        (lower, upper)
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        let n = self.n;
        self.n = 0;
        if n > 0 {
            // nth(n) skips n+1
            if self.iter.nth(n - 1).is_none() {
                return try { init };
            }
        }
        self.iter.try_fold(init, fold)
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if self.n > 0 {
            // nth(n) skips n+1
            if self.iter.nth(self.n - 1).is_none() {
                return init;
            }
        }
        self.iter.fold(init, fold)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Skip<I> where I: ExactSizeIterator {}

#[stable(feature = "double_ended_skip_iterator", since = "1.9.0")]
impl<I> DoubleEndedIterator for Skip<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len() > 0 { self.iter.next_back() } else { None }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<I::Item> {
        let len = self.len();
        if n < len {
            self.iter.nth_back(n)
        } else {
            if len > 0 {
                // consume the original iterator
                self.iter.nth_back(len - 1);
            }
            None
        }
    }

    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        fn check<T, Acc, R: Try<Ok = Acc>>(
            mut n: usize,
            mut fold: impl FnMut(Acc, T) -> R,
        ) -> impl FnMut(Acc, T) -> ControlFlow<R, Acc> {
            move |acc, x| {
                n -= 1;
                let r = fold(acc, x);
                if n == 0 { ControlFlow::Break(r) } else { ControlFlow::from_try(r) }
            }
        }

        let n = self.len();
        if n == 0 { try { init } } else { self.iter.try_rfold(init, check(n, fold)).into_try() }
    }

    fn rfold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn ok<Acc, T>(mut f: impl FnMut(Acc, T) -> Acc) -> impl FnMut(Acc, T) -> Result<Acc, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_rfold(init, ok(fold)).unwrap()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Skip<I> where I: FusedIterator {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: Iterator> SourceIter for Skip<I>
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
unsafe impl<I: InPlaceIterable> InPlaceIterable for Skip<I> {}

/// An iterator that only iterates over the first `n` iterations of `iter`.
///
/// This `struct` is created by the [`take`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`take`]: Iterator::take
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Take<I> {
    pub(super) iter: I,
    pub(super) n: usize,
}
impl<I> Take<I> {
    pub(super) fn new(iter: I, n: usize) -> Take<I> {
        Take { iter, n }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Take<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        if self.n != 0 {
            self.n -= 1;
            self.iter.next()
        } else {
            None
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        if self.n > n {
            self.n -= n + 1;
            self.iter.nth(n)
        } else {
            if self.n > 0 {
                self.iter.nth(self.n - 1);
                self.n = 0;
            }
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.n == 0 {
            return (0, Some(0));
        }

        let (lower, upper) = self.iter.size_hint();

        let lower = cmp::min(lower, self.n);

        let upper = match upper {
            Some(x) if x < self.n => Some(x),
            _ => Some(self.n),
        };

        (lower, upper)
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        fn check<'a, T, Acc, R: Try<Ok = Acc>>(
            n: &'a mut usize,
            mut fold: impl FnMut(Acc, T) -> R + 'a,
        ) -> impl FnMut(Acc, T) -> ControlFlow<R, Acc> + 'a {
            move |acc, x| {
                *n -= 1;
                let r = fold(acc, x);
                if *n == 0 { ControlFlow::Break(r) } else { ControlFlow::from_try(r) }
            }
        }

        if self.n == 0 {
            try { init }
        } else {
            let n = &mut self.n;
            self.iter.try_fold(init, check(n, fold)).into_try()
        }
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(fold)).unwrap()
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: Iterator> SourceIter for Take<I>
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
unsafe impl<I: InPlaceIterable> InPlaceIterable for Take<I> {}

#[stable(feature = "double_ended_take_iterator", since = "1.38.0")]
impl<I> DoubleEndedIterator for Take<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            None
        } else {
            let n = self.n;
            self.n -= 1;
            self.iter.nth_back(self.iter.len().saturating_sub(n))
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.iter.len();
        if self.n > n {
            let m = len.saturating_sub(self.n) + n;
            self.n -= n + 1;
            self.iter.nth_back(m)
        } else {
            if len > 0 {
                self.iter.nth_back(len - 1);
            }
            None
        }
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if self.n == 0 {
            try { init }
        } else {
            let len = self.iter.len();
            if len > self.n && self.iter.nth_back(len - self.n - 1).is_none() {
                try { init }
            } else {
                self.iter.try_rfold(init, fold)
            }
        }
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if self.n == 0 {
            init
        } else {
            let len = self.iter.len();
            if len > self.n && self.iter.nth_back(len - self.n - 1).is_none() {
                init
            } else {
                self.iter.rfold(init, fold)
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Take<I> where I: ExactSizeIterator {}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Take<I> where I: FusedIterator {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<I: TrustedLen> TrustedLen for Take<I> {}

/// An iterator to maintain state while iterating another iterator.
///
/// This `struct` is created by the [`scan`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`scan`]: Iterator::scan
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Scan<I, St, F> {
    iter: I,
    f: F,
    state: St,
}
impl<I, St, F> Scan<I, St, F> {
    pub(super) fn new(iter: I, state: St, f: F) -> Scan<I, St, F> {
        Scan { iter, state, f }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, St: fmt::Debug, F> fmt::Debug for Scan<I, St, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scan").field("iter", &self.iter).field("state", &self.state).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<B, I, St, F> Iterator for Scan<I, St, F>
where
    I: Iterator,
    F: FnMut(&mut St, I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        let a = self.iter.next()?;
        (self.f)(&mut self.state, a)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the scan function
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        fn scan<'a, T, St, B, Acc, R: Try<Ok = Acc>>(
            state: &'a mut St,
            f: &'a mut impl FnMut(&mut St, T) -> Option<B>,
            mut fold: impl FnMut(Acc, B) -> R + 'a,
        ) -> impl FnMut(Acc, T) -> ControlFlow<R, Acc> + 'a {
            move |acc, x| match f(state, x) {
                None => ControlFlow::Break(try { acc }),
                Some(x) => ControlFlow::from_try(fold(acc, x)),
            }
        }

        let state = &mut self.state;
        let f = &mut self.f;
        self.iter.try_fold(init, scan(state, f, fold)).into_try()
    }

    #[inline]
    fn fold<Acc, Fold>(mut self, init: Acc, fold: Fold) -> Acc
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(fold)).unwrap()
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<St, F, B, S: Iterator, I: Iterator> SourceIter for Scan<I, St, F>
where
    I: SourceIter<Source = S>,
    F: FnMut(&mut St, I::Item) -> Option<B>,
{
    type Source = S;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut S {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<St, F, B, I: InPlaceIterable> InPlaceIterable for Scan<I, St, F> where
    F: FnMut(&mut St, I::Item) -> Option<B>
{
}

/// An iterator that calls a function with a reference to each element before
/// yielding it.
///
/// This `struct` is created by the [`inspect`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`inspect`]: Iterator::inspect
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone)]
pub struct Inspect<I, F> {
    iter: I,
    f: F,
}
impl<I, F> Inspect<I, F> {
    pub(super) fn new(iter: I, f: F) -> Inspect<I, F> {
        Inspect { iter, f }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<I: fmt::Debug, F> fmt::Debug for Inspect<I, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Inspect").field("iter", &self.iter).finish()
    }
}

impl<I: Iterator, F> Inspect<I, F>
where
    F: FnMut(&I::Item),
{
    #[inline]
    fn do_inspect(&mut self, elt: Option<I::Item>) -> Option<I::Item> {
        if let Some(ref a) = elt {
            (self.f)(a);
        }

        elt
    }
}

fn inspect_fold<T, Acc>(
    mut f: impl FnMut(&T),
    mut fold: impl FnMut(Acc, T) -> Acc,
) -> impl FnMut(Acc, T) -> Acc {
    move |acc, item| {
        f(&item);
        fold(acc, item)
    }
}

fn inspect_try_fold<'a, T, Acc, R>(
    f: &'a mut impl FnMut(&T),
    mut fold: impl FnMut(Acc, T) -> R + 'a,
) -> impl FnMut(Acc, T) -> R + 'a {
    move |acc, item| {
        f(&item);
        fold(acc, item)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator, F> Iterator for Inspect<I, F>
where
    F: FnMut(&I::Item),
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        let next = self.iter.next();
        self.do_inspect(next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_fold(init, inspect_try_fold(&mut self.f, fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, inspect_fold(self.f, fold))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator, F> DoubleEndedIterator for Inspect<I, F>
where
    F: FnMut(&I::Item),
{
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        let next = self.iter.next_back();
        self.do_inspect(next)
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.iter.try_rfold(init, inspect_try_fold(&mut self.f, fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, inspect_fold(self.f, fold))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator, F> ExactSizeIterator for Inspect<I, F>
where
    F: FnMut(&I::Item),
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator, F> FusedIterator for Inspect<I, F> where F: FnMut(&I::Item) {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I: Iterator, F> SourceIter for Inspect<I, F>
where
    F: FnMut(&I::Item),
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
unsafe impl<I: InPlaceIterable, F> InPlaceIterable for Inspect<I, F> where F: FnMut(&I::Item) {}

/// An iterator adapter that produces output as long as the underlying
/// iterator produces `Result::Ok` values.
///
/// If an error is encountered, the iterator stops and the error is
/// stored.
pub(crate) struct ResultShunt<'a, I, E> {
    iter: I,
    error: &'a mut Result<(), E>,
}

/// Process the given iterator as if it yielded a `T` instead of a
/// `Result<T, _>`. Any errors will stop the inner iterator and
/// the overall result will be an error.
pub(crate) fn process_results<I, T, E, F, U>(iter: I, mut f: F) -> Result<U, E>
where
    I: Iterator<Item = Result<T, E>>,
    for<'a> F: FnMut(ResultShunt<'a, I, E>) -> U,
{
    let mut error = Ok(());
    let shunt = ResultShunt { iter, error: &mut error };
    let value = f(shunt);
    error.map(|()| value)
}

impl<I, T, E> Iterator for ResultShunt<'_, I, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.find(|_| true)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.error.is_err() {
            (0, Some(0))
        } else {
            let (_, upper) = self.iter.size_hint();
            (0, upper)
        }
    }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        let error = &mut *self.error;
        self.iter
            .try_fold(init, |acc, x| match x {
                Ok(x) => ControlFlow::from_try(f(acc, x)),
                Err(e) => {
                    *error = Err(e);
                    ControlFlow::Break(try { acc })
                }
            })
            .into_try()
    }

    fn fold<B, F>(mut self, init: B, fold: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(fold)).unwrap()
    }
}
