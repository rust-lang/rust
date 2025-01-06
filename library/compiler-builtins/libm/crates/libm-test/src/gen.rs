//! Different generators that can create random or systematic bit patterns.

pub mod domain_logspace;
pub mod edge_cases;
pub mod extensive;
pub mod random;

/// A wrapper to turn any iterator into an `ExactSizeIterator`. Asserts the final result to ensure
/// the provided size was correct.
#[derive(Debug)]
pub struct KnownSize<I> {
    total: u64,
    current: u64,
    iter: I,
}

impl<I> KnownSize<I> {
    pub fn new(iter: I, total: u64) -> Self {
        Self { total, current: 0, iter }
    }
}

impl<I: Iterator> Iterator for KnownSize<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.iter.next();
        if next.is_some() {
            self.current += 1;
            return next;
        }

        assert_eq!(self.current, self.total, "total items did not match expected");
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.total - self.current).unwrap();
        (remaining, Some(remaining))
    }
}

impl<I: Iterator> ExactSizeIterator for KnownSize<I> {}
