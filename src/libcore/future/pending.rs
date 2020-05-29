use crate::future::Future;
use crate::marker;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// Creates a future which never resolves, representing a computation that never
/// finishes.
///
/// This `struct` is created by the [`pending`] function. See its
/// documentation for more.
///
/// [`pending`]: fn.pending.html
#[unstable(feature = "future_readiness_fns", issue = "70921")]
#[derive(Debug)]
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct Pending<T> {
    _data: marker::PhantomData<T>,
}

/// Creates a future which never resolves, representing a computation that never
/// finishes.
///
/// # Examples
///
/// ```no_run
/// #![feature(future_readiness_fns)]
/// use core::future;
///
/// # async fn run() {
/// let future = future::pending();
/// let () = future.await;
/// unreachable!();
/// # }
/// ```
#[unstable(feature = "future_readiness_fns", issue = "70921")]
pub fn pending<T>() -> Pending<T> {
    Pending { _data: marker::PhantomData }
}

#[unstable(feature = "future_readiness_fns", issue = "70921")]
impl<T> Future for Pending<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<T> {
        Poll::Pending
    }
}

#[unstable(feature = "future_readiness_fns", issue = "70921")]
impl<T> Unpin for Pending<T> {}

#[unstable(feature = "future_readiness_fns", issue = "70921")]
impl<T> Clone for Pending<T> {
    fn clone(&self) -> Self {
        pending()
    }
}
