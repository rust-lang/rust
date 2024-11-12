#![stable(feature = "wake_trait", since = "1.51.0")]

//! Types and Traits for working with asynchronous tasks.
//!
//! **Note**: Some of the types in this module are only available
//! on platforms that support atomic loads and stores of pointers.
//! This may be detected at compile time using
//! `#[cfg(target_has_atomic = "ptr")]`.

use core::mem::ManuallyDrop;
#[cfg(target_has_atomic = "ptr")]
use core::task::Waker;
use core::task::{LocalWaker, RawWaker, RawWakerVTable};

use crate::rc::Rc;
#[cfg(target_has_atomic = "ptr")]
use crate::sync::Arc;

/// The implementation of waking a task on an executor.
///
/// This trait can be used to create a [`Waker`]. An executor can define an
/// implementation of this trait, and use that to construct a [`Waker`] to pass
/// to the tasks that are executed on that executor.
///
/// This trait is a memory-safe and ergonomic alternative to constructing a
/// [`RawWaker`]. It supports the common executor design in which the data used
/// to wake up a task is stored in an [`Arc`]. Some executors (especially
/// those for embedded systems) cannot use this API, which is why [`RawWaker`]
/// exists as an alternative for those systems.
///
/// To construct a [`Waker`] from some type `W` implementing this trait,
/// wrap it in an [`Arc<W>`](Arc) and call `Waker::from()` on that.
/// It is also possible to convert to [`RawWaker`] in the same way.
///
/// <!-- Ideally we'd link to the `From` impl, but rustdoc doesn't generate any page for it within
///      `alloc` because `alloc` neither defines nor re-exports `From` or `Waker`, and we can't
///      link ../../std/task/struct.Waker.html#impl-From%3CArc%3CW,+Global%3E%3E-for-Waker
///      without getting a link-checking error in CI. -->
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
#[cfg(target_has_atomic = "ptr")]
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
#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "wake_trait", since = "1.51.0")]
impl<W: Wake + Send + Sync + 'static> From<Arc<W>> for Waker {
    /// Use a [`Wake`]-able type as a `Waker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Arc<W>) -> Waker {
        // SAFETY: This is safe because raw_waker safely constructs
        // a RawWaker from Arc<W>.
        unsafe { Waker::from_raw(raw_waker(waker)) }
    }
}
#[cfg(target_has_atomic = "ptr")]
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
#[cfg(target_has_atomic = "ptr")]
#[inline(always)]
fn raw_waker<W: Wake + Send + Sync + 'static>(waker: Arc<W>) -> RawWaker {
    // Increment the reference count of the arc to clone it.
    //
    // The #[inline(always)] is to ensure that raw_waker and clone_waker are
    // always generated in the same code generation unit as one another, and
    // therefore that the structurally identical const-promoted RawWakerVTable
    // within both functions is deduplicated at LLVM IR code generation time.
    // This allows optimizing Waker::will_wake to a single pointer comparison of
    // the vtable pointers, rather than comparing all four function pointers
    // within the vtables.
    #[inline(always)]
    unsafe fn clone_waker<W: Wake + Send + Sync + 'static>(waker: *const ()) -> RawWaker {
        unsafe { Arc::increment_strong_count(waker as *const W) };
        RawWaker::new(
            waker,
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

/// An analogous trait to `Wake` but used to construct a `LocalWaker`.
///
/// This API works in exactly the same way as `Wake`,
/// except that it uses an `Rc` instead of an `Arc`,
/// and the result is a `LocalWaker` instead of a `Waker`.
///
/// The benefits of using `LocalWaker` over `Waker` are that it allows the local waker
/// to hold data that does not implement `Send` and `Sync`. Additionally, it saves calls
/// to `Arc::clone`, which requires atomic synchronization.
///
///
/// # Examples
///
/// This is a simplified example of a `spawn` and a `block_on` function. The `spawn` function
/// is used to push new tasks onto the run queue, while the block on function will remove them
/// and poll them. When a task is woken, it will put itself back on the run queue to be polled
/// by the executor.
///
/// **Note:** This example trades correctness for simplicity. A real world example would interleave
/// poll calls with calls to an io reactor to wait for events instead of spinning on a loop.
///
/// ```rust
/// #![feature(local_waker)]
/// #![feature(noop_waker)]
/// use std::task::{LocalWake, ContextBuilder, LocalWaker, Waker};
/// use std::future::Future;
/// use std::pin::Pin;
/// use std::rc::Rc;
/// use std::cell::RefCell;
/// use std::collections::VecDeque;
///
///
/// thread_local! {
///     // A queue containing all tasks ready to do progress
///     static RUN_QUEUE: RefCell<VecDeque<Rc<Task>>> = RefCell::default();
/// }
///
/// type BoxedFuture = Pin<Box<dyn Future<Output = ()>>>;
///
/// struct Task(RefCell<BoxedFuture>);
///
/// impl LocalWake for Task {
///     fn wake(self: Rc<Self>) {
///         RUN_QUEUE.with_borrow_mut(|queue| {
///             queue.push_back(self)
///         })
///     }
/// }
///
/// fn spawn<F>(future: F)
/// where
///     F: Future<Output=()> + 'static + Send + Sync
/// {
///     let task = RefCell::new(Box::pin(future));
///     RUN_QUEUE.with_borrow_mut(|queue| {
///         queue.push_back(Rc::new(Task(task)));
///     });
/// }
///
/// fn block_on<F>(future: F)
/// where
///     F: Future<Output=()> + 'static + Sync + Send
/// {
///     spawn(future);
///     loop {
///         let Some(task) = RUN_QUEUE.with_borrow_mut(|queue| queue.pop_front()) else {
///             // we exit, since there are no more tasks remaining on the queue
///             return;
///         };
///
///         // cast the Rc<Task> into a `LocalWaker`
///         let local_waker: LocalWaker = task.clone().into();
///         // Build the context using `ContextBuilder`
///         let mut cx = ContextBuilder::from_waker(Waker::noop())
///             .local_waker(&local_waker)
///             .build();
///
///         // Poll the task
///         let _ = task.0
///             .borrow_mut()
///             .as_mut()
///             .poll(&mut cx);
///     }
/// }
///
/// block_on(async {
///     println!("hello world");
/// });
/// ```
///
#[unstable(feature = "local_waker", issue = "118959")]
pub trait LocalWake {
    /// Wake this task.
    #[unstable(feature = "local_waker", issue = "118959")]
    fn wake(self: Rc<Self>);

    /// Wake this task without consuming the local waker.
    ///
    /// If an executor supports a cheaper way to wake without consuming the
    /// waker, it should override this method. By default, it clones the
    /// [`Rc`] and calls [`wake`] on the clone.
    ///
    /// [`wake`]: LocalWaker::wake
    #[unstable(feature = "local_waker", issue = "118959")]
    fn wake_by_ref(self: &Rc<Self>) {
        self.clone().wake();
    }
}

#[unstable(feature = "local_waker", issue = "118959")]
impl<W: LocalWake + 'static> From<Rc<W>> for LocalWaker {
    /// Use a `Wake`-able type as a `LocalWaker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Rc<W>) -> LocalWaker {
        // SAFETY: This is safe because raw_waker safely constructs
        // a RawWaker from Rc<W>.
        unsafe { LocalWaker::from_raw(local_raw_waker(waker)) }
    }
}
#[allow(ineffective_unstable_trait_impl)]
#[unstable(feature = "local_waker", issue = "118959")]
impl<W: LocalWake + 'static> From<Rc<W>> for RawWaker {
    /// Use a `Wake`-able type as a `RawWaker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Rc<W>) -> RawWaker {
        local_raw_waker(waker)
    }
}

// NB: This private function for constructing a RawWaker is used, rather than
// inlining this into the `From<Rc<W>> for RawWaker` impl, to ensure that
// the safety of `From<Rc<W>> for Waker` does not depend on the correct
// trait dispatch - instead both impls call this function directly and
// explicitly.
#[inline(always)]
fn local_raw_waker<W: LocalWake + 'static>(waker: Rc<W>) -> RawWaker {
    // Increment the reference count of the Rc to clone it.
    //
    // Refer to the comment on raw_waker's clone_waker regarding why this is
    // always inline.
    #[inline(always)]
    unsafe fn clone_waker<W: LocalWake + 'static>(waker: *const ()) -> RawWaker {
        unsafe { Rc::increment_strong_count(waker as *const W) };
        RawWaker::new(
            waker,
            &RawWakerVTable::new(clone_waker::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>),
        )
    }

    // Wake by value, moving the Rc into the LocalWake::wake function
    unsafe fn wake<W: LocalWake + 'static>(waker: *const ()) {
        let waker = unsafe { Rc::from_raw(waker as *const W) };
        <W as LocalWake>::wake(waker);
    }

    // Wake by reference, wrap the waker in ManuallyDrop to avoid dropping it
    unsafe fn wake_by_ref<W: LocalWake + 'static>(waker: *const ()) {
        let waker = unsafe { ManuallyDrop::new(Rc::from_raw(waker as *const W)) };
        <W as LocalWake>::wake_by_ref(&waker);
    }

    // Decrement the reference count of the Rc on drop
    unsafe fn drop_waker<W: LocalWake + 'static>(waker: *const ()) {
        unsafe { Rc::decrement_strong_count(waker as *const W) };
    }

    RawWaker::new(
        Rc::into_raw(waker) as *const (),
        &RawWakerVTable::new(clone_waker::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>),
    )
}
