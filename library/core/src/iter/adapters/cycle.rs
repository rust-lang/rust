use crate::iter::FusedIterator;
use crate::num::NonZero;
use crate::ops::Try;

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
    inner: CycleInner<I>,
}

#[derive(Clone, Debug)]
enum CycleInner<I> {
    Exhausted,
    /// `iter` is only `None` at construction, before the first call to `.next` (or other method).
    /// If `iter` is `Some`, then it has yielded at least one element (except if we just put a new clone
    /// in it and are about to call `.next()` on it or similar).
    Running {
        orig: I,
        iter: Option<I>,
    },
}

#[rustc_const_unstable(feature = "const_iter", issue = "92476")]
const impl<I: [const] Clone> Cycle<I> {
    pub(in crate::iter) fn new(iter: I) -> Cycle<I> {
        Cycle { inner: CycleInner::Running { orig: iter, iter: None } }
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
        match &mut self.inner {
            CycleInner::Exhausted => None,
            CycleInner::Running { orig, iter: iter @ None } => {
                let iter = iter.insert(orig.clone());
                match iter.next() {
                    None => {
                        // This is a fresh clone and it is empty;
                        // we exhaust `self` to uphold `FusedIterator`.
                        self.inner = CycleInner::Exhausted;
                        None
                    }
                    y => y,
                }
            }
            CycleInner::Running { orig, iter: Some(iter) } => {
                match iter.next() {
                    None => {
                        // This was not a fresh clone (it already yielded at least one element),
                        // so we replace it with a fresh clone and try again with that one.
                        *iter = orig.clone();
                        let ret = iter.next();
                        if ret.is_none() {
                            // This is a fresh clone and it is empty;
                            // we exhaust `self` to uphold `FusedIterator`.
                            self.inner = CycleInner::Exhausted;
                        }
                        ret
                    }
                    y => y,
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // The cycle iterator is either empty or infinite for "normal" uses.
        // For non-"normal" uses where `orig`'s size hint lies or changes after
        // a clone, the cycle iterator's size hint could be wrong,
        // but there's not much to be done about that.
        match &self.inner {
            CycleInner::Exhausted => (0, Some(0)),
            CycleInner::Running { orig, iter: _ } => match orig.size_hint() {
                sz @ (0, Some(0)) => sz,
                (0, _) => (0, None),
                _ => (usize::MAX, None),
            },
        }
    }

    #[inline]
    fn try_fold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        let CycleInner::Running { orig, iter } = &mut self.inner else {
            return try { acc };
        };
        let iter = if let Some(iter) = iter {
            // Fully iterate the current iterator. This is necessary because
            // `self.iter` may have been fully consumed already, even though we know it wasn't
            // originally empty.
            acc = iter.try_fold(acc, &mut f)?;
            *iter = orig.clone();
            iter
        } else {
            iter.insert(orig.clone())
        };

        // The cloned iterator might not behave the same each time, so we need to stop
        // at the first empty clone to prevent an infinite loop and to uphold `FusedIterator`.
        // We now have a fresh clone in `iter`, so we need to exhaust `self` if it is empty.
        loop {
            let mut is_empty = true;
            acc = iter.try_fold(acc, |acc, x| {
                is_empty = false;
                f(acc, x)
            })?;

            if is_empty {
                // This is a fresh clone and it is empty;
                // we exhaust `self` to uphold `FusedIterator`.
                self.inner = CycleInner::Exhausted;
                return try { acc };
            }

            // The fresh clone was non-empty, but we exhausted it. Clone a new one.
            *iter = orig.clone();
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let Some(mut n) = NonZero::new(n) else {
            return Ok(());
        };

        let CycleInner::Running { orig, iter } = &mut self.inner else {
            return Err(n);
        };

        // If we have a current iterator, use it first. Since it already yielded something,
        // we don't exhaust `self` if it was empty.
        // If we just exhausted the current iterator, or we didn't have a current iterator,
        // clone a new one.
        let iter = match iter {
            Some(iter) => match iter.advance_by(n.get()) {
                Ok(()) => return Ok(()),
                Err(rem) => {
                    *iter = orig.clone();
                    n = rem;
                    iter
                }
            },
            None => iter.insert(orig.clone()),
        };

        // We now have a fresh clone in `iter`, so we need to exhaust `self` if it is empty.
        loop {
            n = match iter.advance_by(n.get()) {
                Ok(()) => return Ok(()),
                e @ Err(rem) if rem == n => {
                    // This is a fresh clone and it is empty;
                    // we exhaust `self` to uphold `FusedIterator`.
                    self.inner = CycleInner::Exhausted;
                    return e;
                }
                Err(rem) => rem,
            };

            // The fresh clone was non-empty, but we exhausted it. Clone a new one.
            *iter = orig.clone();
        }
    }

    // No `fold` override, because `fold` doesn't make much sense for `Cycle`,
    // and we can't do anything better than the default.
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Cycle<I> where I: Clone + Iterator {}
