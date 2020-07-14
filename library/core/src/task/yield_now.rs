use crate::future::Future;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// Cooperatively gives up a timeslice to the executor.
///
/// Calling this function calls move the currently executing future to the back
/// of the execution queue, making room for other futures to execute. This is
/// especially useful after running CPU-intensive operations inside a future.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// #![feature(task_yield_now)]
/// # async fn run() {
/// #
/// use core::task;
///
/// task::yield_now().await;
/// #
/// # }
/// ```
#[unstable(feature = "task_yield_now", issue = "74331")]
#[inline]
pub fn yield_now() -> YieldNow {
    YieldNow { is_polled: false }
}

/// Creates a future that yields back to the executor exactly once.
///
/// This `struct` is created by the [`yield_now`] function. See its
/// documentation for more.
#[unstable(feature = "task_yield_now", issue = "74331")]
#[must_use = "futures do nothing unless you `.await` or poll them"]
#[derive(Debug)]
pub struct YieldNow {
    is_polled: bool,
}

#[unstable(feature = "task_yield_now", issue = "74331")]
impl Future for YieldNow {
    type Output = ();

    // Most futures executors are implemented as a FIFO queue, so all this
    // future does is re-schedule the future back to the end of the queue,
    // giving room for other futures to progress.
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_polled {
            return Poll::Ready(());
        }

        self.is_polled = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}
