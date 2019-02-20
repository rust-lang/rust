#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

use marker::Unpin;
use ops;
use pin::Pin;
use task::{Poll, Waker};

/// A future represents an asynchronous computation.
///
/// A future is a value that may not have finished computing yet. This kind of
/// "asynchronous value" makes it possible for a thread to continue doing useful
/// work while it waits for the value to become available.
///
/// # The `poll` method
///
/// The core method of future, `poll`, *attempts* to resolve the future into a
/// final value. This method does not block if the value is not ready. Instead,
/// the current task is scheduled to be woken up when it's possible to make
/// further progress by `poll`ing again. The wake up is performed using
/// the `waker` argument of the `poll()` method, which is a handle for waking
/// up the current task.
///
/// When using a future, you generally won't call `poll` directly, but instead
/// `await!` the value.
#[must_use = "futures do nothing unless polled"]
pub trait Future {
    /// The type of value produced on completion.
    type Output;

    /// Attempt to resolve the future to a final value, registering
    /// the current task for wakeup if the value is not yet available.
    ///
    /// # Return value
    ///
    /// This function returns:
    ///
    /// - [`Poll::Pending`] if the future is not ready yet
    /// - [`Poll::Ready(val)`] with the result `val` of this future if it
    ///   finished successfully.
    ///
    /// Once a future has finished, clients should not `poll` it again.
    ///
    /// When a future is not ready yet, `poll` returns `Poll::Pending` and
    /// stores a clone of the [`Waker`] to be woken once the future can
    /// make progress. For example, a future waiting for a socket to become
    /// readable would call `.clone()` on the [`Waker`] and store it.
    /// When a signal arrives elsewhere indicating that the socket is readable,
    /// `[Waker::wake]` is called and the socket future's task is awoken.
    /// Once a task has been woken up, it should attempt to `poll` the future
    /// again, which may or may not produce a final value.
    ///
    /// Note that on multiple calls to `poll`, only the most recent
    /// [`Waker`] passed to `poll` should be scheduled to receive a
    /// wakeup.
    ///
    /// # Runtime characteristics
    ///
    /// Futures alone are *inert*; they must be *actively* `poll`ed to make
    /// progress, meaning that each time the current task is woken up, it should
    /// actively re-`poll` pending futures that it still has an interest in.
    ///
    /// The `poll` function is not called repeatedly in a tight loop -- instead,
    /// it should only be called when the future indicates that it is ready to
    /// make progress (by calling `wake()`). If you're familiar with the
    /// `poll(2)` or `select(2)` syscalls on Unix it's worth noting that futures
    /// typically do *not* suffer the same problems of "all wakeups must poll
    /// all events"; they are more like `epoll(4)`.
    ///
    /// An implementation of `poll` should strive to return quickly, and should
    /// not block. Returning quickly prevents unnecessarily clogging up
    /// threads or event loops. If it is known ahead of time that a call to
    /// `poll` may end up taking awhile, the work should be offloaded to a
    /// thread pool (or something similar) to ensure that `poll` can return
    /// quickly.
    ///
    /// An implementation of `poll` may also never cause memory unsafety.
    ///
    /// # Panics
    ///
    /// Once a future has completed (returned `Ready` from `poll`),
    /// then any future calls to `poll` may panic, block forever, or otherwise
    /// cause any kind of bad behavior except causing memory unsafety.
    /// The `Future` trait itself provides no guarantees about the behavior
    /// of `poll` after a future has completed.
    ///
    /// [`Poll::Pending`]: ../task/enum.Poll.html#variant.Pending
    /// [`Poll::Ready(val)`]: ../task/enum.Poll.html#variant.Ready
    /// [`Waker`]: ../task/struct.Waker.html
    /// [`Waker::wake`]: ../task/struct.Waker.html#method.wake
    fn poll(self: Pin<&mut Self>, waker: &Waker) -> Poll<Self::Output>;
}

impl<F: ?Sized + Future + Unpin> Future for &mut F {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, waker: &Waker) -> Poll<Self::Output> {
        F::poll(Pin::new(&mut **self), waker)
    }
}

impl<P> Future for Pin<P>
where
    P: Unpin + ops::DerefMut,
    P::Target: Future,
{
    type Output = <<P as ops::Deref>::Target as Future>::Output;

    fn poll(self: Pin<&mut Self>, waker: &Waker) -> Poll<Self::Output> {
        Pin::get_mut(self).as_mut().poll(waker)
    }
}
