use crate::fmt;
use crate::iter::InPlaceIterable;
use crate::iter::adapters::SourceIter;
use crate::num::NonZero;
use crate::ops::{ControlFlow, Try};

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
    pub(in crate::iter) fn new(iter: I, state: St, f: F) -> Scan<I, St, F> {
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
        R: Try<Output = Acc>,
    {
        fn scan<'a, T, St, B, Acc, R: Try<Output = Acc>>(
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

    impl_fold_via_try_fold! { fold -> try_fold }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<St, F, I> SourceIter for Scan<I, St, F>
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
unsafe impl<St, F, I: InPlaceIterable> InPlaceIterable for Scan<I, St, F> {
    const EXPAND_BY: Option<NonZero<usize>> = I::EXPAND_BY;
    const MERGE_BY: Option<NonZero<usize>> = I::MERGE_BY;
}
