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
    Running { orig: I, iter: I },
}

#[rustc_const_unstable(feature = "const_iter", issue = "92476")]
const impl<I: [const] Clone> Cycle<I> {
    pub(in crate::iter) fn new(iter: I) -> Cycle<I> {
        Cycle { inner: CycleInner::Running { orig: iter.clone(), iter } }
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
            CycleInner::Running { orig, iter } => {
                match iter.next() {
                    None => {
                        *iter = orig.clone();
                        let ret = iter.next();
                        if ret.is_none() {
                            // If we found an empty clone,
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
        // fully iterate the current iterator. this is necessary because
        // `self.iter` may have been fully consumed already, even if it wasn't
        // originally empty.
        acc = iter.try_fold(acc, &mut f)?;
        *iter = orig.clone();

        // The cloned iterator might not behave the same each time, so we need to stop
        // at the first empty clone to prevent an infinite loop and to uphold `FusedIterator`.
        loop {
            let mut is_empty = true;
            acc = iter.try_fold(acc, |acc, x| {
                is_empty = false;
                f(acc, x)
            })?;

            if is_empty {
                self.inner = CycleInner::Exhausted;
                return try { acc };
            }
            *iter = orig.clone();
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let CycleInner::Running { orig, iter } = &mut self.inner else {
            return NonZero::new(n).map_or(Ok(()), Err);
        };

        let mut n = match iter.advance_by(n) {
            Ok(()) => return Ok(()),
            Err(rem) => rem.get(),
        };

        while n > 0 {
            *iter = orig.clone();
            n = match iter.advance_by(n) {
                Ok(()) => return Ok(()),
                e @ Err(rem) if rem.get() == n => {
                    // If we found an empty clone,
                    // we exhaust `self` to uphold `FusedIterator`.
                    self.inner = CycleInner::Exhausted;
                    return e;
                }
                Err(rem) => rem.get(),
            };
        }

        debug_assert_eq!(n, 0, "while loop should only end when n == 0");

        Ok(())
    }

    // No `fold` override, because `fold` doesn't make much sense for `Cycle`,
    // and we can't do anything better than the default.
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Cycle<I> where I: Clone + Iterator {}
