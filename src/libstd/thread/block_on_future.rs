//! Future/Task/Thread integration.
//! The method defined in this module allows to block a thread until an
//! async task completes.

use crate::future::Future;
use crate::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use crate::mem;
use crate::pin::Pin;
use crate::sync::Arc;
use super::{current, park, Inner, Thread};

/// Carries a flag that is used to wakeup the executor.
/// A pointer to this struct is passed to the thread-local waker.
struct LocalWakeState {
    is_woken: bool,
    waker_was_cloned: bool,
}

/// Returns the vtable that is used for waking up the executor
/// from another thread.
fn threadsafe_waker_vtable() -> &'static RawWakerVTable {
    &RawWakerVTable::new(
        clone_threadsafe_waker,
        wake_threadsafe_waker,
        wake_threadsafe_waker_by_ref,
        drop_threadsafe_waker,
    )
}

/// Returns the vtable that is used for waking up the executor
/// from inside it's execution on the current thread.
fn current_thread_waker_vtable() -> &'static RawWakerVTable {
    &RawWakerVTable::new(
        create_threadsafe_waker,
        wake_current_thread,
        wake_current_thread_by_ref,
        |_| {},
    )
}

/// This method will be called when the waker reference gets cloned,
/// which makes it possible to transfer it to another thread. In this
/// case we have to create a threadsafe `Waker`. In order to to this
/// we retain the thread handle and store it in the new `RawWaker`s
/// data pointer.
unsafe fn create_threadsafe_waker(data: *const()) -> RawWaker {
    let wake_state = data as *mut LocalWakeState;
    (*wake_state).waker_was_cloned = true;

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
    RawWaker::new(ptr, threadsafe_waker_vtable())
}

unsafe fn clone_threadsafe_waker(data: *const()) -> RawWaker {
    increase_refcount(data);
    RawWaker::new(data, threadsafe_waker_vtable())
}

fn wake_current_thread(_data: *const()) {
    unreachable!("A current thread waker can only be woken by reference");
}

unsafe fn wake_current_thread_by_ref(data: *const()) {
    let wake_state = data as *mut LocalWakeState;
    (*wake_state).is_woken = true;
}

unsafe fn wake_threadsafe_waker(data: *const ()) {
    let arc_thread_inner = Arc::from_raw(data as *const Inner);
    let thread = Thread {
        inner: arc_thread_inner,
    };
    thread.unpark();
}

unsafe fn wake_threadsafe_waker_by_ref(data: *const ()) {
    // Retain `Arc`, but don't touch refcount by wrapping in `ManuallyDrop`
    let arc_thread_inner = Arc::from_raw(data as *const Inner);
    let thread = mem::ManuallyDrop::new(Thread {
        inner: arc_thread_inner,
    });
    thread.unpark();
}

unsafe fn drop_threadsafe_waker(data: *const ()) {
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

    let mut waker_state = LocalWakeState {
        is_woken: true,
        waker_was_cloned: false,
    };

    // Safety: The `Waker` that we create here is references data on the current
    // callstack. This is safe, since the polled `Future` only gets a reference
    // to this `Waker`. When it tries to clone the `Waker`, a threadsafe and owned
    // version is created instead.
    unsafe {
        let waker = Waker::from_raw(RawWaker::new(
            &waker_state as *const LocalWakeState as *const (),
            current_thread_waker_vtable()));

        let mut cx = Context::from_waker(&waker);
        loop {
            while waker_state.is_woken {
                // Reset is_woken, so that we do not spin if the poll does not
                // directly wake us up.
                waker_state.is_woken = false;
                if let Poll::Ready(task_result) = future.as_mut().poll(&mut cx) {
                    return task_result;
                }
            }

            // The task is not ready, and the `Waker` had not been woken from the
            // current thread. In order for us to proceed we wait until the
            // thread gets unparked by another thread. If the `Waker` has not been
            // cloned this will never happen and represents a deadlock, which
            // gets reported here.
            if !waker_state.waker_was_cloned {
                panic!("Deadlock: Task is not ready, but the Waker had not been cloned");
                // Note: This flag is never reset, since a `Waker` that had been cloned
                // once can be cloned more often to wakeup this executor. We don't
                // have knowledge on how many clones are around - therefore the
                // deadlock detection only works for the case the `Waker` never
                // gets cloned.
            }
            park();
            // If thread::park has returned, we have been notified by another
            // thread. Therefore we are woken.
            // Remark: This flag can not be set by the other thread directly,
            // because it may no longer be alive at the point of time when
            // wake() is called.
            waker_state.is_woken = true;
        }
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

        let waker_state = LocalWakeState {
            is_woken: true,
            waker_was_cloned: false,
        };

        let waker = unsafe { Waker::from_raw(RawWaker::new(
            &waker_state as *const LocalWakeState as *const (),
            current_thread_waker_vtable())) };

        waker.wake_by_ref();
        assert_eq!(original, current_thread_refcount());

        let clone1 = waker.clone();
        assert_eq!(original + 1, current_thread_refcount());
        let clone2 = waker.clone();
        assert_eq!(original + 2, current_thread_refcount());
        let clone3 = clone1.clone();
        assert_eq!(original + 3, current_thread_refcount());

        drop(clone1);
        assert_eq!(original + 2, current_thread_refcount());

        clone2.wake_by_ref();
        assert_eq!(original + 2, current_thread_refcount());
        clone2.wake();
        assert_eq!(original + 1, current_thread_refcount());

        clone3.wake_by_ref();
        assert_eq!(original + 1, current_thread_refcount());
        clone3.wake();
        assert_eq!(original, current_thread_refcount());
    }
}
