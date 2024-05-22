use crate::array;
use crate::iter::adapters::SourceIter;
use crate::iter::{
    ByRefSized, FusedIterator, InPlaceIterable, TrustedFused, TrustedRandomAccessNoCoerce,
};
use crate::num::NonZero;
use crate::ops::{ControlFlow, NeverShortCircuit, Try};

/// An iterator over `N` elements of the iterator at a time.
///
/// The chunks do not overlap. If `N` does not divide the length of the
/// iterator, then the last up to `N-1` elements will be omitted.
///
/// This `struct` is created by the [`array_chunks`][Iterator::array_chunks]
/// method on [`Iterator`]. See its documentation for more.
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
pub struct ArrayChunks<I: Iterator, const N: usize> {
    iter: I,
    remainder: Option<array::IntoIter<I::Item, N>>,
}

impl<I, const N: usize> ArrayChunks<I, N>
where
    I: Iterator,
{
    #[track_caller]
    pub(in crate::iter) fn new(iter: I) -> Self {
        assert!(N != 0, "chunk size must be non-zero");
        Self { iter, remainder: None }
    }

    /// Returns an iterator over the remaining elements of the original iterator
    /// that are not going to be returned by this iterator. The returned
    /// iterator will yield at most `N-1` elements.
    ///
    /// # Example
    /// ```
    /// # // Also serves as a regression test for https://github.com/rust-lang/rust/issues/123333
    /// # #![feature(iter_array_chunks)]
    /// let x = [1,2,3,4,5].into_iter().array_chunks::<2>();
    /// let mut rem = x.into_remainder().unwrap();
    /// assert_eq!(rem.next(), Some(5));
    /// assert_eq!(rem.next(), None);
    /// ```
    #[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
    #[inline]
    pub fn into_remainder(mut self) -> Option<array::IntoIter<I::Item, N>> {
        if self.remainder.is_none() {
            while let Some(_) = self.next() {}
        }
        self.remainder
    }
}

#[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
impl<I, const N: usize> Iterator for ArrayChunks<I, N>
where
    I: Iterator,
{
    type Item = [I::Item; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.try_for_each(ControlFlow::Break).break_value()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        (lower / N, upper.map(|n| n / N))
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count() / N
    }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let mut acc = init;
        loop {
            match self.iter.next_chunk() {
                Ok(chunk) => acc = f(acc, chunk)?,
                Err(remainder) => {
                    // Make sure to not override `self.remainder` with an empty array
                    // when `next` is called after `ArrayChunks` exhaustion.
                    self.remainder.get_or_insert(remainder);

                    break try { acc };
                }
            }
        }
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        <Self as SpecFold>::fold(self, init, f)
    }
}

#[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
impl<I, const N: usize> DoubleEndedIterator for ArrayChunks<I, N>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.try_rfold((), |(), x| ControlFlow::Break(x)).break_value()
    }

    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        // We are iterating from the back we need to first handle the remainder.
        self.next_back_remainder();

        let mut acc = init;
        let mut iter = ByRefSized(&mut self.iter).rev();

        // NB remainder is handled by `next_back_remainder`, so
        // `next_chunk` can't return `Err` with non-empty remainder
        // (assuming correct `I as ExactSizeIterator` impl).
        while let Ok(mut chunk) = iter.next_chunk() {
            // FIXME: do not do double reverse
            //        (we could instead add `next_chunk_back` for example)
            chunk.reverse();
            acc = f(acc, chunk)?
        }

        try { acc }
    }

    impl_fold_via_try_fold! { rfold -> try_rfold }
}

impl<I, const N: usize> ArrayChunks<I, N>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    /// Updates `self.remainder` such that `self.iter.len` is divisible by `N`.
    fn next_back_remainder(&mut self) {
        // Make sure to not override `self.remainder` with an empty array
        // when `next_back` is called after `ArrayChunks` exhaustion.
        if self.remainder.is_some() {
            return;
        }

        // We use the `ExactSizeIterator` implementation of the underlying
        // iterator to know how many remaining elements there are.
        let rem = self.iter.len() % N;

        // Take the last `rem` elements out of `self.iter`.
        let mut remainder =
            // SAFETY: `unwrap_err` always succeeds because x % N < N for all x.
            unsafe { self.iter.by_ref().rev().take(rem).next_chunk().unwrap_err_unchecked() };

        // We used `.rev()` above, so we need to re-reverse the reminder
        remainder.as_mut_slice().reverse();
        self.remainder = Some(remainder);
    }
}

#[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
impl<I, const N: usize> FusedIterator for ArrayChunks<I, N> where I: FusedIterator {}

#[unstable(issue = "none", feature = "trusted_fused")]
unsafe impl<I, const N: usize> TrustedFused for ArrayChunks<I, N> where I: TrustedFused + Iterator {}

#[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
impl<I, const N: usize> ExactSizeIterator for ArrayChunks<I, N>
where
    I: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len() / N
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.iter.len() < N
    }
}

trait SpecFold: Iterator {
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B;
}

impl<I, const N: usize> SpecFold for ArrayChunks<I, N>
where
    I: Iterator,
{
    #[inline]
    default fn fold<B, F>(mut self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.try_fold(init, NeverShortCircuit::wrap_mut_2(f)).0
    }
}

impl<I, const N: usize> SpecFold for ArrayChunks<I, N>
where
    I: Iterator + TrustedRandomAccessNoCoerce,
{
    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        let inner_len = self.iter.size();
        let mut i = 0;
        // Use a while loop because (0..len).step_by(N) doesn't optimize well.
        while inner_len - i >= N {
            let chunk = crate::array::from_fn(|local| {
                // SAFETY: The method consumes the iterator and the loop condition ensures that
                // all accesses are in bounds and only happen once.
                unsafe {
                    let idx = i + local;
                    self.iter.__iterator_get_unchecked(idx)
                }
            });
            accum = f(accum, chunk);
            i += N;
        }

        // unlike try_fold this method does not need to take care of the remainder
        // since `self` will be dropped

        accum
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, const N: usize> SourceIter for ArrayChunks<I, N>
where
    I: SourceIter + Iterator,
{
    type Source = I::Source;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut I::Source {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I: InPlaceIterable + Iterator, const N: usize> InPlaceIterable for ArrayChunks<I, N> {
    const EXPAND_BY: Option<NonZero<usize>> = I::EXPAND_BY;
    const MERGE_BY: Option<NonZero<usize>> = const {
        match (I::MERGE_BY, NonZero::new(N)) {
            (Some(m), Some(n)) => m.checked_mul(n),
            _ => None,
        }
    };
}
