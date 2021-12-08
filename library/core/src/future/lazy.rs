use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

/// A future that lazily executes a closure.
#[derive(Debug, Clone)]
#[must_use = "futures do nothing unless you `.await` or poll them"]
#[unstable(feature = "future_lazy", issue = "91647")]
pub struct Lazy<F>(Option<F>);

#[unstable(feature = "future_lazy", issue = "91647")]
impl<F> Unpin for Lazy<F> {}

/// Creates a new future that lazily executes a closure.
///
/// The provided closure is only run once the future is polled.
///
/// # Examples
///
/// ```
/// #![feature(future_lazy)]
///
/// # let _ = async {
/// use std::future;
///
/// let a = future::lazy(|_| 1);
/// assert_eq!(a.await, 1);
///
/// let b = future::lazy(|_| -> i32 {
///     panic!("oh no!")
/// });
/// drop(b); // closure is never run
/// # };
/// ```
#[unstable(feature = "future_lazy", issue = "91647")]
pub fn lazy<F, R>(f: F) -> Lazy<F>
where
    F: FnOnce(&mut Context<'_>) -> R,
{
    Lazy(Some(f))
}

#[unstable(feature = "future_lazy", issue = "91647")]
impl<F, R> Future for Lazy<F>
where
    F: FnOnce(&mut Context<'_>) -> R,
{
    type Output = R;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        let f = self.0.take().expect("`Lazy` polled after completion");
        Poll::Ready((f)(cx))
    }
}
