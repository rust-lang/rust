#![allow(unused)]

use crate::future::Future;
use crate::pin::Pin;
use crate::task::{Context, Poll};
use crate::{fmt, mem, ptr};

/// A [`Future`] that maps the output of a wrapped [`Future`].
///
/// Returned by [`Future::map`].
#[unstable(feature = "future_map", issue = "none")]
pub struct Map<Fut, F> {
    inner: Option<(Fut, F)>,
}

impl<Fut, F> Map<Fut, F> {
    pub(crate) fn new(future: Fut, f: F) -> Self {
        Self { inner: Some((future, f)) }
    }
}

#[unstable(feature = "future_map", issue = "none")]
impl<Fut: fmt::Debug, F> fmt::Debug for Map<Fut, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Map").field("future", &self.inner.as_ref().map(|(fut, _)| fut)).finish()
    }
}

#[unstable(feature = "future_map", issue = "none")]
impl<Fut: Future, F: FnOnce(Fut::Output) -> U, U> Future for Map<Fut, F> {
    type Output = U;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: we make sure to not move the inner future
        unsafe {
            let this = Pin::into_inner_unchecked(self);
            match &mut this.inner {
                Some((future, _)) => {
                    let pin = Pin::new_unchecked(&mut *future);
                    match pin.poll(cx) {
                        Poll::Ready(value) => {
                            // The future must be dropped in-place since it is pinned.
                            ptr::drop_in_place(future);

                            let (future, map) = this.inner.take().unwrap_unchecked();
                            mem::forget(future);

                            Poll::Ready(map(value))
                        }
                        Poll::Pending => Poll::Pending,
                    }
                }
                None => panic!("Map must not be polled after it returned `Poll::Ready`"),
            }
        }
    }
}
