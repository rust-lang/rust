use crate::ops::Try;

/// Like `Iterator::by_ref`, but requiring `Sized` so it can forward generics.
///
/// Ideally this will no longer be required, eventually, but as can be seen in
/// the benchmarks (as of Feb 2022 at least) `by_ref` can have performance cost.
pub(crate) struct ByRefSized<'a, I>(pub &'a mut I);

impl<I: Iterator> Iterator for ByRefSized<'_, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        self.0.advance_by(n)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.0.fold(init, f)
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.0.try_fold(init, f)
    }
}
