use crate::fmt;
use crate::iter::Peekable;

/// An iterator that only accepts elements while `predicate` and `peek` returns `Some(_)`.
///
/// This `struct` is created by the [`peekable_map_while`] method on [`Peekable`]. See its
/// documentation for more.
///
/// [`peekable_map_while`]: Peekable::peekable_map_while
/// [`Peekable`]: Peekable
#[unstable(feature = "peekable_map_while", issue = "none")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct PeekableMapWhile<'iter, I: Iterator, P> {
    iter: &'iter mut Peekable<I>,
    predicate: P,
}

#[unstable(feature = "peekable_map_while", issue = "none")]
impl<'iter, I: Iterator, P> PeekableMapWhile<'iter, I, P> {
    pub(in crate::iter) fn new(
        iter: &'iter mut Peekable<I>,
        predicate: P,
    ) -> PeekableMapWhile<'iter, I, P> {
        PeekableMapWhile { iter, predicate }
    }
}

#[unstable(feature = "peekable_map_while", issue = "none")]
impl<'iter, I, P> fmt::Debug for PeekableMapWhile<'iter, I, P>
where
    I: fmt::Debug + Iterator,
    <I as Iterator>::Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeekableMapWhile").field("iter", &self.iter).finish()
    }
}

#[unstable(feature = "peekable_map_while", issue = "none")]
impl<'iter, B, I: Iterator, P> Iterator for PeekableMapWhile<'iter, I, P>
where
    P: FnMut(&I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        let x = self.iter.peek()?;
        if let Some(b) = (self.predicate)(x) {
            self.iter.next();
            Some(b)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper) // can't know a lower bound, due to the predicate
    }
}
