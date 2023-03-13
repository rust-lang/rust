use crate::num::NonZeroUsize;
use crate::ops::{NeverShortCircuit, Try};

/// Like `Iterator::by_ref`, but requiring `Sized` so it can forward generics.
///
/// Ideally this will no longer be required, eventually, but as can be seen in
/// the benchmarks (as of Feb 2022 at least) `by_ref` can have performance cost.
#[unstable(feature = "std_internals", issue = "none")]
#[derive(Debug)]
pub struct ByRefSized<'a, I>(pub &'a mut I);

// The following implementations use UFCS-style, rather than trusting autoderef,
// to avoid accidentally calling the `&mut Iterator` implementations.

#[unstable(feature = "std_internals", issue = "none")]
impl<I: Iterator> Iterator for ByRefSized<'_, I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        I::next(self.0)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        I::size_hint(self.0)
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        I::advance_by(self.0, n)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        I::nth(self.0, n)
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        // `fold` needs ownership, so this can't forward directly.
        I::try_fold(self.0, init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        I::try_fold(self.0, init, f)
    }
}

#[unstable(feature = "std_internals", issue = "none")]
impl<I: DoubleEndedIterator> DoubleEndedIterator for ByRefSized<'_, I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        I::next_back(self.0)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        I::advance_back_by(self.0, n)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        I::nth_back(self.0, n)
    }

    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        // `rfold` needs ownership, so this can't forward directly.
        I::try_rfold(self.0, init, NeverShortCircuit::wrap_mut_2(f)).0
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        I::try_rfold(self.0, init, f)
    }
}
