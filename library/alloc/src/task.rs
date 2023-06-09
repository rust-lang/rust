#![stable(feature = "wake_trait", since = "1.51.0")]

//! Types and Traits for working with asynchronous tasks.
//!
//! **Note**: This module is only available on platforms that support atomic
//! loads and stores of pointers. This may be detected at compile time using
//! `#[cfg(target_has_atomic = "ptr")]`.

use core::mem::ManuallyDrop;
use core::task::{RawWaker, RawWakerVTable, Waker};

use crate::sync::Arc;

/// The implementation of waking a task on an executor.
///
/// This trait can be used to create a [`Waker`]. An executor can define an
/// implementation of this trait, and use that to construct a Waker to pass
/// to the tasks that are executed on that executor.
///
/// This trait is a memory-safe and ergonomic alternative to constructing a
/// [`RawWaker`]. It supports the common executor design in which the data used
/// to wake up a task is stored in an [`Arc`]. Some executors (especially
/// those for embedded systems) cannot use this API, which is why [`RawWaker`]
/// exists as an alternative for those systems.
///
/// [arc]: ../../std/sync/struct.Arc.html
///
/// # Examples
///
/// A basic `block_on` function that takes a future and runs it to completion on
/// the current thread.
///
/// **Note:** This example trades correctness for simplicity. In order to prevent
/// deadlocks, production-grade implementations will also need to handle
/// intermediate calls to `thread::unpark` as well as nested invocations.
///
/// ```rust
/// use std::future::Future;
/// use std::sync::Arc;
/// use std::task::{Context, Poll, Wake};
/// use std::thread::{self, Thread};
/// use core::pin::pin;
///
/// /// A waker that wakes up the current thread when called.
/// struct ThreadWaker(Thread);
///
/// impl Wake for ThreadWaker {
///     fn wake(self: Arc<Self>) {
///         self.0.unpark();
///     }
/// }
///
/// /// Run a future to completion on the current thread.
/// fn block_on<T>(fut: impl Future<Output = T>) -> T {
///     // Pin the future so it can be polled.
///     let mut fut = pin!(fut);
///
///     // Create a new context to be passed to the future.
///     let t = thread::current();
///     let waker = Arc::new(ThreadWaker(t)).into();
///     let mut cx = Context::from_waker(&waker);
///
///     // Run the future to completion.
///     loop {
///         match fut.as_mut().poll(&mut cx) {
///             Poll::Ready(res) => return res,
///             Poll::Pending => thread::park(),
///         }
///     }
/// }
///
/// block_on(async {
///     println!("Hi from inside a future!");
/// });
/// ```
#[stable(feature = "wake_trait", since = "1.51.0")]
pub trait Wake {
    /// Wake this task.
    #[stable(feature = "wake_trait", since = "1.51.0")]
    fn wake(self: Arc<Self>);

    /// Wake this task without consuming the waker.
    ///
    /// If an executor supports a cheaper way to wake without consuming the
    /// waker, it should override this method. By default, it clones the
    /// [`Arc`] and calls [`wake`] on the clone.
    ///
    /// [`wake`]: Wake::wake
    #[stable(feature = "wake_trait", since = "1.51.0")]
    fn wake_by_ref(self: &Arc<Self>) {
        self.clone().wake();
    }
}

#[stable(feature = "wake_trait", since = "1.51.0")]
impl<W: Wake + Send + Sync + 'static> From<Arc<W>> for Waker {
    /// Use a `Wake`-able type as a `Waker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Arc<W>) -> Waker {
        // SAFETY: This is safe because raw_waker safely constructs
        // a RawWaker from Arc<W>.
        unsafe { Waker::from_raw(raw_waker(waker)) }
    }
}

#[stable(feature = "wake_trait", since = "1.51.0")]
impl<W: Wake + Send + Sync + 'static> From<Arc<W>> for RawWaker {
    /// Use a `Wake`-able type as a `RawWaker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Arc<W>) -> RawWaker {
        raw_waker(waker)
    }
}

// NB: This private function for constructing a RawWaker is used, rather than
// inlining this into the `From<Arc<W>> for RawWaker` impl, to ensure that
// the safety of `From<Arc<W>> for Waker` does not depend on the correct
// trait dispatch - instead both impls call this function directly and
// explicitly.
#[inline(always)]
fn raw_waker<W: Wake + Send + Sync + 'static>(waker: Arc<W>) -> RawWaker {
    // Increment the reference count of the arc to clone it.
    unsafe fn clone_waker<W: Wake + Send + Sync + 'static>(waker: *const ()) -> RawWaker {
        unsafe { Arc::increment_strong_count(waker as *const W) };
        RawWaker::new(
            waker as *const (),
            &RawWakerVTable::new(clone_waker::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>),
        )
    }

    // Wake by value, moving the Arc into the Wake::wake function
    unsafe fn wake<W: Wake + Send + Sync + 'static>(waker: *const ()) {
        let waker = unsafe { Arc::from_raw(waker as *const W) };
        <W as Wake>::wake(waker);
    }

    // Wake by reference, wrap the waker in ManuallyDrop to avoid dropping it
    unsafe fn wake_by_ref<W: Wake + Send + Sync + 'static>(waker: *const ()) {
        let waker = unsafe { ManuallyDrop::new(Arc::from_raw(waker as *const W)) };
        <W as Wake>::wake_by_ref(&waker);
    }

    // Decrement the reference count of the Arc on drop
    unsafe fn drop_waker<W: Wake + Send + Sync + 'static>(waker: *const ()) {
        unsafe { Arc::decrement_strong_count(waker as *const W) };
    }

    RawWaker::new(
        Arc::into_raw(waker) as *const (),
        &RawWakerVTable::new(clone_waker::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>),
    )
}
