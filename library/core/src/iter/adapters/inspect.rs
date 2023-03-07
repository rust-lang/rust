use crate::fmt;
use crate::iter::{adapters::SourceIter, FusedIterator, InPlaceIterable};
use crate::ops::Try;

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
    pub(in crate::iter) fn new(iter: I, f: F) -> Inspect<I, F> {
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
        R: Try<Output = Acc>,
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
        R: Try<Output = Acc>,
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
unsafe impl<I, F> SourceIter for Inspect<I, F>
where
    I: SourceIter,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable, F> InPlaceIterable for Inspect<I, F> where F: FnMut(&I::Item) {}
