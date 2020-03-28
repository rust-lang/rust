use crate::convert::Infallible;
use crate::intrinsics;
use crate::iter::{
    DoubleEndedIterator, ExactSizeIterator, FusedIterator, Iterator, TrustedRandomAccess,
};
use crate::ops::Try;

trait Stop: Sized {
    /// Switch internal state of `Fuse` to indicate that underlying iterator returned `None`
    fn stop<I>(state: &mut Result<I, Self>);
}

impl Stop for () {
    fn stop<I>(state: &mut Result<I, Self>) {
        *state = Err(())
    }
}

impl Stop for Infallible {
    #[inline(always)]
    fn stop<I>(_state: &mut Result<I, Self>) {
        // Intentionally does nothing: fused iterator returns `None`s after returning `None`,
        // so there is no need to alter state
    }
}

trait StopState {
    /// Type of value used to indicate that iterator returned `None`
    type State: Stop + crate::fmt::Debug + Clone;
}

impl<I: ?Sized> StopState for I {
    default type State = ();
}

impl<I: FusedIterator + ?Sized> StopState for I {
    type State = Infallible;
}

type StopStateOf<I> = <I as StopState>::State;

/// An iterator that yields `None` forever after the underlying iterator
/// yields `None` once.
///
/// This `struct` is created by the [`fuse`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`fuse`]: trait.Iterator.html#method.fuse
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Fuse<I> {
    iter: Result<I, StopStateOf<I>>,
}

impl<I> Fuse<I> {
    pub(in crate::iter) fn new(iter: I) -> Fuse<I> {
        Fuse { iter: Ok(iter) }
    }

    #[inline(always)]
    fn stop(&mut self) {
        StopStateOf::<I>::stop(&mut self.iter);
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Fuse<I> where I: Iterator {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for Fuse<I>
where
    I: Iterator,
{
    type Item = <I as Iterator>::Item;

    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        let next = self.iter.as_mut().ok()?.next();
        if next.is_none() {
            self.stop();
        }
        next
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        let nth = self.iter.as_mut().ok()?.nth(n);
        if nth.is_none() {
            self.stop();
        }
        nth
    }

    #[inline]
    fn last(self) -> Option<I::Item> {
        self.iter.ok()?.last()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.map_or(0, I::count)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.as_ref().map_or((0, Some(0)), I::size_hint)
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if let Ok(ref mut iter) = self.iter {
            acc = iter.try_fold(acc, fold)?;
            self.stop();
        }
        Try::from_ok(acc)
    }

    #[inline]
    fn fold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Ok(iter) = self.iter {
            acc = iter.fold(acc, fold);
        }
        acc
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        let found = self.iter.as_mut().ok()?.find(predicate);
        if found.is_none() {
            self.stop();
        }
        found
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for Fuse<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<<I as Iterator>::Item> {
        let next = self.iter.as_mut().ok()?.next_back();
        if next.is_none() {
            self.stop();
        }
        next
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<<I as Iterator>::Item> {
        let nth = self.iter.as_mut().ok()?.nth_back(n);
        if nth.is_none() {
            self.stop();
        }
        nth
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, mut acc: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        if let Ok(ref mut iter) = self.iter {
            acc = iter.try_rfold(acc, fold)?;
            self.stop();
        }
        Try::from_ok(acc)
    }

    #[inline]
    fn rfold<Acc, Fold>(self, mut acc: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if let Ok(iter) = self.iter {
            acc = iter.rfold(acc, fold);
        }
        acc
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        let found = self.iter.as_mut().ok()?.rfind(predicate);
        if found.is_none() {
            self.stop();
        }
        found
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> ExactSizeIterator for Fuse<I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iter.as_ref().map_or(0, I::len)
    }

    fn is_empty(&self) -> bool {
        self.iter.as_ref().map_or(true, I::is_empty)
    }
}

unsafe impl<I> TrustedRandomAccess for Fuse<I>
where
    I: TrustedRandomAccess,
{
    unsafe fn get_unchecked(&mut self, i: usize) -> I::Item {
        match self.iter {
            Ok(ref mut iter) => iter.get_unchecked(i),
            // SAFETY: the caller asserts there is an item at `i`, so we're not exhausted.
            Err(_) => intrinsics::unreachable(),
        }
    }

    fn may_have_side_effect() -> bool {
        I::may_have_side_effect()
    }
}
