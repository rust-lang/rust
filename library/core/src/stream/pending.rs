use core::fmt;
use core::marker::PhantomData;
use core::pin::Pin;
use core::stream::Stream;
use core::task::{Context, Poll};

/// Creates a stream that never returns any elements.
///
/// The returned stream will always return `Pending` when polled.
#[unstable(feature = "stream_pending", issue = "91683")]
pub fn pending<T>() -> Pending<T> {
    Pending { _t: PhantomData }
}

/// A stream that never returns any elements.
///
/// This stream is created by the [`pending`] function. See its
/// documentation for more.
#[must_use = "streams do nothing unless polled"]
#[unstable(feature = "stream_pending", issue = "91683")]
pub struct Pending<T> {
    _t: PhantomData<T>,
}

#[unstable(feature = "stream_pending", issue = "91683")]
impl<T> Unpin for Pending<T> {}

#[unstable(feature = "stream_pending", issue = "91683")]
impl<T> Stream for Pending<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

#[unstable(feature = "stream_pending", issue = "91683")]
impl<T> fmt::Debug for Pending<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Pending").finish()
    }
}

#[unstable(feature = "stream_pending", issue = "91683")]
impl<T> Clone for Pending<T> {
    fn clone(&self) -> Self {
        pending()
    }
}
