use crate::pin::Pin;

use crate::async_iter::AsyncIterator;
use crate::task::{Context, Poll};

/// An async iterator that was created from iterator.
///
/// This async iterator is created by the [`from_iter`] function.
/// See it documentation for more.
///
/// [`from_iter`]: fn.from_iter.html
#[unstable(feature = "async_iter_from_iter", issue = "81798")]
#[derive(Clone, Debug)]
pub struct FromIter<I> {
    iter: I,
}

#[unstable(feature = "async_iter_from_iter", issue = "81798")]
impl<I> Unpin for FromIter<I> {}

/// Converts an iterator into an async iterator.
#[unstable(feature = "async_iter_from_iter", issue = "81798")]
pub fn from_iter<I: IntoIterator>(iter: I) -> FromIter<I::IntoIter> {
    FromIter { iter: iter.into_iter() }
}

#[unstable(feature = "async_iter_from_iter", issue = "81798")]
impl<I: Iterator> AsyncIterator for FromIter<I> {
    type Item = I::Item;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
