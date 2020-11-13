use crate::future::Future;
use crate::pin::Pin;
use crate::stream::Stream;
use crate::task::{Context, Poll};

/// A future which advances the stream and returns the next value.
///
/// This `struct` is created by [`Stream::next`]. See its documentation for more.
#[unstable(feature = "async_stream", issue = "79024")]
#[derive(Debug)]
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct Next<'a, S: ?Sized> {
    stream: &'a mut S,
}

impl<'a, S: ?Sized> Next<'a, S> {
    /// Create a new instance of `Next`.
    pub(crate) fn new(stream: &'a mut S) -> Self {
        Self { stream }
    }
}

#[unstable(feature = "async_stream", issue = "79024")]
impl<S: Stream + Unpin + ?Sized> Future for Next<'_, S> {
    type Output = Option<S::Item>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut *self.stream).poll_next(cx)
    }
}
