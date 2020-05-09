use crate::future::Future;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// Creates a future that is immediately ready with a value.
///
/// This `struct` is created by the [`ready`] function. See its
/// documentation for more.
///
/// [`ready`]: fn.ready.html
#[unstable(feature = "future_readiness_fns", issue = "70921")]
#[derive(Debug, Clone)]
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct Ready<T>(Option<T>);

#[unstable(feature = "future_readiness_fns", issue = "70921")]
impl<T> Unpin for Ready<T> {}

#[unstable(feature = "future_readiness_fns", issue = "70921")]
impl<T> Future for Ready<T> {
    type Output = T;

    #[inline]
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<T> {
        Poll::Ready(self.0.take().expect("Ready polled after completion"))
    }
}

/// Creates a future that is immediately ready with a value.
///
/// # Examples
///
/// ```
/// #![feature(future_readiness_fns)]
/// use core::future;
///
/// # async fn run() {
/// let a = future::ready(1);
/// assert_eq!(a.await, 1);
/// # }
/// ```
#[unstable(feature = "future_readiness_fns", issue = "70921")]
pub fn ready<T>(t: T) -> Ready<T> {
    Ready(Some(t))
}
