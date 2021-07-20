use crate::pin::Pin;

use crate::stream::Stream;
use crate::task::{Context, Poll};

/// A stream that was created from iterator.
///
/// This stream is created by the [`from_iter`] function.
/// See it documentation for more.
///
/// [`from_iter`]: fn.from_iter.html
#[unstable(feature = "stream_from_iter", issue = "81798")]
#[derive(Clone, Debug)]
pub struct FromIter<I> {
    iter: I,
}

#[unstable(feature = "stream_from_iter", issue = "81798")]
impl<I> Unpin for FromIter<I> {}

/// Converts an iterator into a stream.
#[unstable(feature = "stream_from_iter", issue = "81798")]
pub fn from_iter<I: IntoIterator>(iter: I) -> FromIter<I::IntoIter> {
    FromIter { iter: iter.into_iter() }
}

#[unstable(feature = "stream_from_iter", issue = "81798")]
impl<I: Iterator> Stream for FromIter<I> {
    type Item = I::Item;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.iter.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
