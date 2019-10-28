//! Future/Task/Thread integration.
//! The method defined in this module allows to block a thread until an
//! async task completes.

use crate::future::Future;
use crate::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use crate::mem;
use crate::pin::Pin;
use crate::sync::Arc;
use super::{current, park, Inner, Thread};

/// Returns the vtable that is used for waking up the executor
/// from any thread.
fn waker_vtable() -> &'static RawWakerVTable {
    &RawWakerVTable::new(
        clone_waker,
        wake_waker,
        wake_waker_by_ref,
        drop_waker,
    )
}

/// Creates a [`RawWaker`] which captures the current thread handle
/// and allows to wake up the [`block_on_future`] executor from any
/// thread by calling [`Thread::unpark()`].
fn create_threadsafe_raw_waker() -> RawWaker {
    // Get the `Arc<Inner>` of a current thread handle and store into in
    // the type erased pointer.
    //
    // This mechanism exploits the fact that `Thread` is already an `Arc<Inner>`.
    // Therefore in order to create clones of the thread handle we can just
    // directly create clones of the Inner state, and recreate the `Thread`
    // handle when necessary.
    //
    // If the implementation of `Thread` would be changed to something different,
    // we would need to wrap the complete `Thread` object in another `Arc` by
    // adopt the following line to:
    // `let arc_thread = Arc::new(current());`
    let arc_thread_inner = current().inner;
    let ptr = Arc::into_raw(arc_thread_inner) as *const ();
    RawWaker::new(ptr, waker_vtable())
}

unsafe fn clone_waker(data: *const()) -> RawWaker {
    increase_refcount(data);
    RawWaker::new(data, waker_vtable())
}

unsafe fn wake_waker(data: *const ()) {
    let arc_thread_inner = Arc::from_raw(data as *const Inner);
    let thread = Thread {
        inner: arc_thread_inner,
    };
    thread.unpark();
}

unsafe fn wake_waker_by_ref(data: *const ()) {
    // Retain `Arc`, but don't touch refcount by wrapping in `ManuallyDrop`
    let arc_thread_inner = Arc::from_raw(data as *const Inner);
    let thread = mem::ManuallyDrop::new(Thread {
        inner: arc_thread_inner,
    });
    thread.unpark();
}

unsafe fn drop_waker(data: *const ()) {
    drop(Thread {
        inner: Arc::from_raw(data as *const Inner),
    })
}

#[allow(clippy::redundant_clone)] // The clone here isn't actually redundant.
unsafe fn increase_refcount(data: *const ()) {
    // Retain Arc, but don't touch refcount by wrapping in `ManuallyDrop`
    let arc = mem::ManuallyDrop::new(Arc::<Inner>::from_raw(data as *const Inner));
    // Now increase refcount, but don't drop new refcount either.
    // Note: `Arc::clone()` will not panic but abort if the newrefcount is
    // unrealistically high. Therefore this is safe, as long not more `Waker`
    // clones are created than what is allowed for other `Arc` instances.
    let _arc_clone: mem::ManuallyDrop<_> = arc.clone();
}

/// Runs a [`Future`] to completion on the current thread and returns its output
/// value.
///
/// This method represents a minimal [`Future`]s executor. The executor will
/// drive the [`Future`] to completion on the current thread. The executor will
/// be providing a [`Waker`] to all polled Child-[`Future`]s which can be woken
/// from either the current thread or any other thread.
///
/// # Examples
///
/// ```
/// #![feature(block_on_future)]
/// use std::thread::block_on_future;
///
/// let result = block_on_future(async {
///     5
/// });
/// assert_eq!(5, result);
/// ```
#[unstable(feature = "block_on_future", issue = "0")]
pub fn block_on_future<F: Future>(mut future: F) -> F::Output {
    // Pin the `Future` - which had been moved into this function - on the stack.
    // Safety: This is safe since the `Future` gets aliased and will not be moved
    // out of this function again.
    let mut future = unsafe { Pin::new_unchecked(&mut future) };

    // Safety: The `Waker` that we create upholds all guarantees that are expected
    // from a `Waker`
    let waker = unsafe {
        Waker::from_raw(create_threadsafe_raw_waker())
    };

    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(task_result) = future.as_mut().poll(&mut cx) {
            return task_result;
        }

        // The task is not ready. In order for us to proceed we wait until the
        // thread gets unparked. If the `Waker` had been woken inside `.poll()`,
        // then `park()` will immediately return, and we will call `.poll()`
        // again without any wait period.
        park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_thread_refcount() -> usize {
        Arc::strong_count(&current().inner)
    }

    /// Check that we don't leak any thread::Inner instances to wrong refcount
    /// handling.
    #[test]
    fn check_refcounts() {
        let original = current_thread_refcount();

        let waker = unsafe { Waker::from_raw(create_threadsafe_raw_waker()) };
        assert_eq!(original + 1, current_thread_refcount());

        waker.wake_by_ref();
        assert_eq!(original + 1, current_thread_refcount());

        let clone1 = waker.clone();
        assert_eq!(original + 2, current_thread_refcount());
        let clone2 = waker.clone();
        assert_eq!(original + 3, current_thread_refcount());
        let clone3 = clone1.clone();
        assert_eq!(original + 4, current_thread_refcount());

        drop(clone1);
        assert_eq!(original + 3, current_thread_refcount());

        clone2.wake_by_ref();
        assert_eq!(original + 3, current_thread_refcount());
        clone2.wake();
        assert_eq!(original + 2, current_thread_refcount());

        drop(waker);
        assert_eq!(original + 1, current_thread_refcount());

        clone3.wake_by_ref();
        assert_eq!(original + 1, current_thread_refcount());
        clone3.wake();
        assert_eq!(original, current_thread_refcount());
    }
}
