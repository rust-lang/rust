use crate::iter::{FusedIterator, Peekable};

/// An iterator that maps the values of `iter` with `f`.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "peek_map", issue = "118474")]
#[derive(Debug)]
pub struct PeekMap<P, F> {
    pub(crate) peekable: P,
    f: F,
}

impl<I: Iterator, F> PeekMap<Peekable<I>, F> {
    pub(in crate::iter) fn new(peekable: Peekable<I>, f: F) -> PeekMap<Peekable<I>, F> {
        PeekMap { peekable, f }
    }
}

#[unstable(feature = "peek_map", issue = "118474")]
impl<B, I: Iterator, F> Iterator for PeekMap<Peekable<I>, F>
where
    F: FnMut(I::Item, Option<&I::Item>) -> B,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        Some((&mut self.f)(self.peekable.next()?, self.peekable.peek()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.peekable.size_hint()
    }
}
#[unstable(feature = "peek_map", issue = "118474")]
impl<B, I: ExactSizeIterator, F> ExactSizeIterator for PeekMap<Peekable<I>, F>
where
    F: FnMut(I::Item, Option<&I::Item>) -> B,
{
    fn len(&self) -> usize {
        self.peekable.len()
    }

    fn is_empty(&self) -> bool {
        self.peekable.is_empty()
    }
}
#[unstable(feature = "peek_map", issue = "118474")]
impl<B, I: FusedIterator, F> FusedIterator for PeekMap<Peekable<I>, F> where
    F: FnMut(I::Item, Option<&I::Item>) -> B
{
}
