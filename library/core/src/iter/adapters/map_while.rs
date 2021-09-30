use crate::fmt;
use crate::iter::{adapters::SourceIter, InPlaceIterable};
use crate::ops::{ControlFlow, Try};

/// An iterator that only accepts elements while `predicate` returns `Some(_)`.
///
/// This `struct` is created by the [`map_while`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`map_while`]: Iterator::map_while
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iter_map_while", since = "1.57.0")]
#[derive(Clone)]
pub struct MapWhile<I, P> {
    iter: I,
    predicate: P,
}

impl<I, P> MapWhile<I, P> {
    pub(in crate::iter) fn new(iter: I, predicate: P) -> MapWhile<I, P> {
        MapWhile { iter, predicate }
    }
}

#[stable(feature = "iter_map_while", since = "1.57.0")]
impl<I: fmt::Debug, P> fmt::Debug for MapWhile<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapWhile").field("iter", &self.iter).finish()
    }
}

#[stable(feature = "iter_map_while", since = "1.57.0")]
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
        R: Try<Output = Acc>,
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
unsafe impl<I, P> SourceIter for MapWhile<I, P>
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
unsafe impl<B, I: InPlaceIterable, P> InPlaceIterable for MapWhile<I, P> where
    P: FnMut(I::Item) -> Option<B>
{
}
