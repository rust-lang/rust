use crate::fmt;
use crate::iter::{adapters::SourceIter, FusedIterator, InPlaceIterable};
use crate::ops::Try;

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
    pub(in crate::iter) fn new(iter: I, predicate: P) -> Filter<I, P> {
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
