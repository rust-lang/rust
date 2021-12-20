use crate::fmt;
use crate::future::Future;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// Resolves to the output of polling a future once.
///
/// This `struct` is created by [`poll_once()`]. See its
/// documentation for more.
#[unstable(feature = "future_poll_once", issue = "92115")]
pub struct PollOnce<F> {
    pub(crate) future: F,
}

#[unstable(feature = "future_poll_once", issue = "92115")]
impl<F> fmt::Debug for PollOnce<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PollOnce").finish()
    }
}

#[unstable(feature = "future_poll_once", issue = "92115")]
impl<F> Future for PollOnce<F>
where
    F: Future + Unpin,
{
    type Output = Poll<F::Output>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(Pin::new(&mut self.future).poll(cx))
    }
}
