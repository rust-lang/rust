#![allow(unused)]

use crate::fmt;
use crate::future::Future;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// A [`Future`] that maps the output of a wrapped [`Future`].
///
/// Returned by [`Future::map`].
#[unstable(feature = "future_map", issue = "none")]
pub struct Map<Fut, F> {
    future: Option<Fut>,
    f: Option<F>,
}

impl<Fut, F> Map<Fut, F> {
    pub(crate) fn new(future: Fut, f: F) -> Self {
        Self { future: Some(future), f: Some(f) }
    }
}

#[unstable(feature = "future_map", issue = "none")]
impl<Fut: fmt::Debug, F> fmt::Debug for Map<Fut, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Map").field("future", &self.future).finish()
    }
}

#[unstable(feature = "future_map", issue = "none")]
impl<Fut: Future, F: FnOnce(Fut::Output) -> U, U> Future for Map<Fut, F> {
    type Output = U;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}
