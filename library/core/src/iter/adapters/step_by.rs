use crate::{intrinsics, iter::from_fn, ops::Try};

/// An iterator for stepping iterators by a custom amount.
///
/// This `struct` is created by the [`step_by`] method on [`Iterator`]. See
/// its documentation for more.
///
/// [`step_by`]: Iterator::step_by
/// [`Iterator`]: trait.Iterator.html
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "iterator_step_by", since = "1.28.0")]
#[derive(Clone, Debug)]
pub struct StepBy<I> {
    iter: I,
    step: usize,
    first_take: bool,
}

impl<I> StepBy<I> {
    pub(in crate::iter) fn new(iter: I, step: usize) -> StepBy<I> {
        assert!(step != 0);
        StepBy { iter, step: step - 1, first_take: true }
    }
}

#[stable(feature = "iterator_step_by", since = "1.28.0")]
impl<I> Iterator for StepBy<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            self.iter.next()
        } else {
            self.iter.nth(self.step)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        #[inline]
        fn first_size(step: usize) -> impl Fn(usize) -> usize {
            move |n| if n == 0 { 0 } else { 1 + (n - 1) / (step + 1) }
        }

        #[inline]
        fn other_size(step: usize) -> impl Fn(usize) -> usize {
            move |n| n / (step + 1)
        }

        let (low, high) = self.iter.size_hint();

        if self.first_take {
            let f = first_size(self.step);
            (f(low), high.map(f))
        } else {
            let f = other_size(self.step);
            (f(low), high.map(f))
        }
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            let first = self.iter.next();
            if n == 0 {
                return first;
            }
            n -= 1;
        }
        // n and self.step are indices, we need to add 1 to get the amount of elements
        // When calling `.nth`, we need to subtract 1 again to convert back to an index
        // step + 1 can't overflow because `.step_by` sets `self.step` to `step - 1`
        let mut step = self.step + 1;
        // n + 1 could overflow
        // thus, if n is usize::MAX, instead of adding one, we call .nth(step)
        if n == usize::MAX {
            self.iter.nth(step - 1);
        } else {
            n += 1;
        }

        // overflow handling
        loop {
            let mul = n.checked_mul(step);
            {
                if intrinsics::likely(mul.is_some()) {
                    return self.iter.nth(mul.unwrap() - 1);
                }
            }
            let div_n = usize::MAX / n;
            let div_step = usize::MAX / step;
            let nth_n = div_n * n;
            let nth_step = div_step * step;
            let nth = if nth_n > nth_step {
                step -= div_n;
                nth_n
            } else {
                n -= div_step;
                nth_step
            };
            self.iter.nth(nth - 1);
        }
    }

    fn try_fold<Acc, F, R>(&mut self, mut acc: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn nth<I: Iterator>(iter: &mut I, step: usize) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth(step)
        }

        if self.first_take {
            self.first_take = false;
            match self.iter.next() {
                None => return try { acc },
                Some(x) => acc = f(acc, x)?,
            }
        }
        from_fn(nth(&mut self.iter, self.step)).try_fold(acc, f)
    }

    fn fold<Acc, F>(mut self, mut acc: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn nth<I: Iterator>(iter: &mut I, step: usize) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth(step)
        }

        if self.first_take {
            self.first_take = false;
            match self.iter.next() {
                None => return acc,
                Some(x) => acc = f(acc, x),
            }
        }
        from_fn(nth(&mut self.iter, self.step)).fold(acc, f)
    }
}

impl<I> StepBy<I>
where
    I: ExactSizeIterator,
{
    // The zero-based index starting from the end of the iterator of the
    // last element. Used in the `DoubleEndedIterator` implementation.
    fn next_back_index(&self) -> usize {
        let rem = self.iter.len() % (self.step + 1);
        if self.first_take {
            if rem == 0 { self.step } else { rem - 1 }
        } else {
            rem
        }
    }
}

#[stable(feature = "double_ended_step_by_iterator", since = "1.38.0")]
impl<I> DoubleEndedIterator for StepBy<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.nth_back(self.next_back_index())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // `self.iter.nth_back(usize::MAX)` does the right thing here when `n`
        // is out of bounds because the length of `self.iter` does not exceed
        // `usize::MAX` (because `I: ExactSizeIterator`) and `nth_back` is
        // zero-indexed
        let n = n.saturating_mul(self.step + 1).saturating_add(self.next_back_index());
        self.iter.nth_back(n)
    }

    fn try_rfold<Acc, F, R>(&mut self, init: Acc, mut f: F) -> R
    where
        F: FnMut(Acc, Self::Item) -> R,
        R: Try<Ok = Acc>,
    {
        #[inline]
        fn nth_back<I: DoubleEndedIterator>(
            iter: &mut I,
            step: usize,
        ) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth_back(step)
        }

        match self.next_back() {
            None => try { init },
            Some(x) => {
                let acc = f(init, x)?;
                from_fn(nth_back(&mut self.iter, self.step)).try_fold(acc, f)
            }
        }
    }

    #[inline]
    fn rfold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        Self: Sized,
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        #[inline]
        fn nth_back<I: DoubleEndedIterator>(
            iter: &mut I,
            step: usize,
        ) -> impl FnMut() -> Option<I::Item> + '_ {
            move || iter.nth_back(step)
        }

        match self.next_back() {
            None => init,
            Some(x) => {
                let acc = f(init, x);
                from_fn(nth_back(&mut self.iter, self.step)).fold(acc, f)
            }
        }
    }
}

// StepBy can only make the iterator shorter, so the len will still fit.
#[stable(feature = "iterator_step_by", since = "1.28.0")]
impl<I> ExactSizeIterator for StepBy<I> where I: ExactSizeIterator {}
