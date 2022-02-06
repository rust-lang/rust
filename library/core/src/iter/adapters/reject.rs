use crate::fmt;
use crate::iter::{adapters::SourceIter, FusedIterator, InPlaceIterable};
use crate::ops::Try;

/// FIXME: add docs
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "option_iter_reject", issue = "none")]
#[derive(Clone)]
pub struct Reject<I, P> {
    iter: I,
    predicate: P,
}

impl<I, P> Reject<I, P> {
    pub(in crate::iter) fn new(iter: I, predicate: P) -> Reject<I, P> {
        Reject { iter, predicate }
    }
}

#[unstable(feature = "option_iter_reject", issue = "none")]
impl<I: fmt::Debug, P> fmt::Debug for Reject<I, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reject").field("iter", &self.iter).finish()
    }
}

fn reject_fold<T, Acc>(
    mut predicate: impl FnMut(&T) -> bool,
    mut fold: impl FnMut(Acc, T) -> Acc,
) -> impl FnMut(Acc, T) -> Acc {
    move |acc, item| if !predicate(&item) { fold(acc, item) } else { acc }
}

fn reject_try_fold<'a, T, Acc, R: Try<Output = Acc>>(
    predicate: &'a mut impl FnMut(&T) -> bool,
    mut fold: impl FnMut(Acc, T) -> R + 'a,
) -> impl FnMut(Acc, T) -> R + 'a {
    move |acc, item| if !predicate(&item) { fold(acc, item) } else { try { acc } }
}

#[unstable(feature = "option_iter_reject", issue = "none")]
impl<I: Iterator, P> Iterator for Reject<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        self.iter.find(|v| !(self.predicate)(v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }

    // this special case allows the compiler to make `.reject(_).count()`
    // branchless. Barring perfect branch prediction (which is unattainable in
    // the general case), this will be much faster in >90% of cases (containing
    // virtually all real workloads) and only a tiny bit slower in the rest.
    //
    // Having this specialization thus allows us to write `.reject(p).count()`
    // where we would otherwise write `.map(|x| !p(x) as usize).sum()`, which is
    // less readable and also less backwards-compatible to Rust before 1.10.
    //
    // Using the branchless version will also simplify the LLVM byte code, thus
    // leaving more budget for LLVM optimizations.
    #[inline]
    fn count(self) -> usize {
        #[inline]
        fn to_usize<T>(mut predicate: impl FnMut(&T) -> bool) -> impl FnMut(T) -> usize {
            move |x| !predicate(&x) as usize
        }

        self.iter.map(to_usize(self.predicate)).sum()
    }

    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.iter.try_fold(init, reject_try_fold(&mut self.predicate, fold))
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.fold(init, reject_fold(self.predicate, fold))
    }
}

#[unstable(feature = "option_iter_reject", issue = "none")]
impl<I: DoubleEndedIterator, P> DoubleEndedIterator for Reject<I, P>
where
    P: FnMut(&I::Item) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        self.iter.rfind(|v| !(self.predicate)(v))
    }

    #[inline]
    fn try_rfold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        self.iter.try_rfold(init, reject_try_fold(&mut self.predicate, fold))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.iter.rfold(init, reject_fold(self.predicate, fold))
    }
}

#[unstable(feature = "option_iter_reject", issue = "none")]
impl<I: FusedIterator, P> FusedIterator for Reject<I, P> where P: FnMut(&I::Item) -> bool {}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<P, I> SourceIter for Reject<I, P>
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
unsafe impl<I: InPlaceIterable, P> InPlaceIterable for Reject<I, P> where P: FnMut(&I::Item) -> bool {}
