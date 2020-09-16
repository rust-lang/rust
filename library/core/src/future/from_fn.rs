use crate::fmt;
use crate::future::Future;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// Creates a future that wraps a function returning `Poll`.
///
/// Polling the future delegates to the wrapped function.
///
/// # Examples
///
/// ```
/// #![feature(future_from_fn)]
/// # async fn run() {
/// use core::future;
/// use core::task::{Context, Poll};
///
/// let fut = future::from_fn(|_cx: &mut Context<'_>| -> Poll<String> {
///     Poll::Ready("Hello, World!".into())
/// });
///
/// assert_eq!(fut.await, "Hello, World!".to_owned());
/// # };
/// ```
#[unstable(feature = "future_from_fn", issue = "72302")]
pub fn from_fn<T, F>(f: F) -> FromFn<F>
where
    F: FnMut(&mut Context<'_>) -> Poll<T>,
{
    FromFn { f }
}

/// A Future that wraps a function returning `Poll`.
///
/// This `struct` is created by [`from_fn()`]. See its
/// documentation for more.
#[must_use = "futures do nothing unless you `.await` or poll them"]
#[unstable(feature = "future_from_fn", issue = "72302")]
pub struct FromFn<F> {
    f: F,
}

#[unstable(feature = "future_from_fn", issue = "72302")]
impl<F> Unpin for FromFn<F> {}

#[unstable(feature = "future_from_fn", issue = "72302")]
impl<F> fmt::Debug for FromFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FromFn").finish()
    }
}

#[unstable(feature = "future_from_fn", issue = "72302")]
impl<T, F> Future for FromFn<F>
where
    F: FnMut(&mut Context<'_>) -> Poll<T>,
{
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        (&mut self.f)(cx)
    }
}
