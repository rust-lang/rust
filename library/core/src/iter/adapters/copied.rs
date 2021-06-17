use crate::iter::adapters::{zip::try_get_unchecked, TrustedRandomAccess};
use crate::iter::{FusedIterator, TrustedLen};
use crate::ops::Try;

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
    pub(in crate::iter) fn new(it: I) -> Copied<I> {
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
        R: Try<Output = B>,
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

    #[doc(hidden)]
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
        R: Try<Output = B>,
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
    const MAY_HAVE_SIDE_EFFECT: bool = I::MAY_HAVE_SIDE_EFFECT;
}

#[stable(feature = "iter_copied", since = "1.36.0")]
unsafe impl<'a, I, T: 'a> TrustedLen for Copied<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Copy,
{
}
