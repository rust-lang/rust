use crate::{iter::FusedIterator, ops::Try};

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
    orig: I,
    iter: I,
}

impl<I: Clone> Cycle<I> {
    pub(in crate::iter) fn new(iter: I) -> Cycle<I> {
        Cycle { orig: iter.clone(), iter }
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
        match self.iter.next() {
            None => {
                self.iter = self.orig.clone();
                self.iter.next()
            }
            y => y,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // the cycle iterator is either empty or infinite
        match self.orig.size_hint() {
            sz @ (0, Some(0)) => sz,
            (0, _) => (0, None),
            _ => (usize::MAX, None),
        }
    }

    #[inline]
    fn try_fold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        // fully iterate the current iterator. this is necessary because
        // `self.iter` may be empty even when `self.orig` isn't
        acc = self.iter.try_fold(acc, &mut f)?;
        self.iter = self.orig.clone();

        // complete a full cycle, keeping track of whether the cycled
        // iterator is empty or not. we need to return early in case
        // of an empty iterator to prevent an infinite loop
        let mut is_empty = true;
        acc = self.iter.try_fold(acc, |acc, x| {
            is_empty = false;
            f(acc, x)
        })?;

        if is_empty {
            return try { acc };
        }

        loop {
            self.iter = self.orig.clone();
            acc = self.iter.try_fold(acc, &mut f)?;
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        let mut rem = n;
        match self.iter.advance_by(rem) {
            ret @ Ok(_) => return ret,
            Err(advanced) => rem -= advanced,
        }

        while rem > 0 {
            self.iter = self.orig.clone();
            match self.iter.advance_by(rem) {
                ret @ Ok(_) => return ret,
                Err(0) => return Err(n - rem),
                Err(advanced) => rem -= advanced,
            }
        }

        Ok(())
    }

    // No `fold` override, because `fold` doesn't make much sense for `Cycle`,
    // and we can't do anything better than the default.
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I> FusedIterator for Cycle<I> where I: Clone + Iterator {}
