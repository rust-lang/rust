use crate::iter::{FusedIterator, Peekable};

/// An iterator that maps the values of `iter` with `f`.
///
/// This struct is created by the [`peek_map`] method on [`Peekable`]. See its
/// documentation for more.
///
/// [`peek_map`]: Peekable::peek_map
/// [`Peekable`]: struct.Peekable.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// #![feature(peek_map)]
///
/// let a = [1, 2, 3];
/// let mut iter = a.into_iter().peekable().peek_map(|x, next| x * *next.unwrap_or(&1));
///
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(6));
/// assert_eq!(iter.next(), Some(3));
/// assert_eq!(iter.next(), None);
/// ```
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "peek_map", issue = "118474")]
#[derive(Debug)]
pub struct PeekMap<T, F> {
    pub(crate) t: T,
    f: F,
}

impl<I: Iterator, F> PeekMap<Peekable<I>, F> {
    pub(in crate::iter) fn new(peekable: Peekable<I>, f: F) -> PeekMap<Peekable<I>, F> {
        PeekMap { t: peekable, f }
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
        Some((&mut self.f)(self.t.next()?, self.t.peek()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.t.size_hint()
    }
}
#[unstable(feature = "peek_map", issue = "118474")]
impl<B, I: ExactSizeIterator, F> ExactSizeIterator for PeekMap<Peekable<I>, F>
where
    F: FnMut(I::Item, Option<&I::Item>) -> B,
{
    fn len(&self) -> usize {
        self.t.len()
    }

    fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}
#[unstable(feature = "peek_map", issue = "118474")]
impl<B, I: FusedIterator, F> FusedIterator for PeekMap<Peekable<I>, F> where
    F: FnMut(I::Item, Option<&I::Item>) -> B
{
}
