use crate::iter::{DoubleEndedIterator, Fuse, FusedIterator, Iterator, TrustedLen};
use crate::ops::Try;
use crate::usize;

/// An iterator that links two iterators together, in a chain.
///
/// This `struct` is created by the [`chain`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`chain`]: trait.Iterator.html#method.chain
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chain<A, B> {
    // These are `Fuse`d so we don't have to manually track which part is already exhausted.
    // However, the `Defuse` wrapper hides the `FusedIterator` implementation, so we don't
    // use the specialized `Fuse` that unconditionally descends into the iterator, because
    // that could be expensive to keep revisiting stuff like nested chains.
    a: Fuse<Defuse<A>>,
    b: Fuse<Defuse<B>>,
}
impl<A, B> Chain<A, B> {
    pub(in super::super) fn new(a: A, b: B) -> Chain<A, B> {
        Chain { a: Fuse::new(Defuse(a)), b: Fuse::new(Defuse(b)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Chain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        match self.a.next() {
            None => self.b.next(),
            item => item,
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn count(self) -> usize {
        self.a.count() + self.b.count()
    }

    fn try_fold<Acc, F, R>(&mut self, init: Acc, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        let accum = self.a.try_fold(init, &mut f)?;
        self.b.try_fold(accum, f)
    }

    fn fold<Acc, F>(self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.a.fold(init, &mut f);
        self.b.fold(accum, f)
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<A::Item> {
        for x in self.a.by_ref() {
            if n == 0 {
                return Some(x);
            }
            n -= 1;
        }
        self.b.nth(n)
    }

    #[inline]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        match self.a.find(&mut predicate) {
            None => self.b.find(predicate),
            item => item,
        }
    }

    #[inline]
    fn last(self) -> Option<A::Item> {
        // Must exhaust a before b.
        let a_last = self.a.last();
        let b_last = self.b.last();
        b_last.or(a_last)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = a_lower.saturating_add(b_lower);

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) => x.checked_add(y),
            _ => None,
        };

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Chain<A, B>
where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item = A::Item>,
{
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        match self.b.next_back() {
            None => self.a.next_back(),
            item => item,
        }
    }

    #[inline]
    fn nth_back(&mut self, mut n: usize) -> Option<A::Item> {
        for x in self.b.by_ref().rev() {
            if n == 0 {
                return Some(x);
            }
            n -= 1;
        }
        self.a.nth_back(n)
    }

    #[inline]
    fn rfind<P>(&mut self, mut predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        match self.b.rfind(&mut predicate) {
            None => self.a.rfind(predicate),
            item => item,
        }
    }

    fn try_rfold<Acc, F, R>(&mut self, init: Acc, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        let accum = self.b.try_rfold(init, &mut f)?;
        self.a.try_rfold(accum, f)
    }

    fn rfold<Acc, F>(self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let accum = self.b.rfold(init, &mut f);
        self.a.rfold(accum, f)
    }
}

// Note: *both* must be fused to handle double-ended iterators.
// Now that we `Fuse` both sides, we *could* implement this unconditionally,
// but we should be cautious about committing to that in the public API.
#[stable(feature = "fused", since = "1.26.0")]
impl<A, B> FusedIterator for Chain<A, B>
where
    A: FusedIterator,
    B: FusedIterator<Item = A::Item>,
{
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A, B> TrustedLen for Chain<A, B>
where
    A: TrustedLen,
    B: TrustedLen<Item = A::Item>,
{
}

/// Wrapper that forwards everything but `FusedIterator`.
#[derive(Clone, Debug)]
struct Defuse<I>(I);

impl<I: Iterator> Iterator for Defuse<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        self.0.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn try_fold<Acc, F, R>(&mut self, init: Acc, f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, I::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.0.try_fold(init, f)
    }

    #[inline]
    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, I::Item) -> Acc,
    {
        self.0.fold(init, f)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        self.0.nth(n)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<I::Item>
    where
        P: FnMut(&I::Item) -> bool,
    {
        self.0.find(predicate)
    }

    #[inline]
    fn last(self) -> Option<I::Item> {
        self.0.last()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for Defuse<I> {
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<I::Item> {
        self.0.nth_back(n)
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<I::Item>
    where
        P: FnMut(&I::Item) -> bool,
    {
        self.0.rfind(predicate)
    }

    #[inline]
    fn try_rfold<Acc, F, R>(&mut self, init: Acc, f: F) -> R
    where
        Self: Sized,
        F: FnMut(Acc, I::Item) -> R,
        R: Try<Ok = Acc>,
    {
        self.0.try_rfold(init, f)
    }

    #[inline]
    fn rfold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, I::Item) -> Acc,
    {
        self.0.rfold(init, f)
    }
}
