#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

use fmt;
use marker::Unpin;
use ptr::NonNull;

/// A `Waker` is a handle for waking up a task by notifying its executor that it
/// is ready to be run.
///
/// This handle contains a trait object pointing to an instance of the `UnsafeWake`
/// trait, allowing notifications to get routed through it.
#[repr(transparent)]
pub struct Waker {
    inner: NonNull<dyn UnsafeWake>,
}

impl Unpin for Waker {}
unsafe impl Send for Waker {}
unsafe impl Sync for Waker {}

impl Waker {
    /// Constructs a new `Waker` directly.
    ///
    /// Note that most code will not need to call this. Implementers of the
    /// `UnsafeWake` trait will typically provide a wrapper that calls this
    /// but you otherwise shouldn't call it directly.
    ///
    /// If you're working with the standard library then it's recommended to
    /// use the `Waker::from` function instead which works with the safe
    /// `Arc` type and the safe `Wake` trait.
    #[inline]
    pub unsafe fn new(inner: NonNull<dyn UnsafeWake>) -> Self {
        Waker { inner }
    }

    /// Wake up the task associated with this `Waker`.
    #[inline]
    pub fn wake(&self) {
        unsafe { self.inner.as_ref().wake() }
    }

    /// Returns `true` if or not this `Waker` and `other` awaken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns `true`, it is guaranteed that the `Waker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake(&self, other: &Waker) -> bool {
        self.inner == other.inner
    }

    /// Returns `true` if or not this `Waker` and `other` `LocalWaker` awaken
    /// the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns true, it is guaranteed that the `Waker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake_local(&self, other: &LocalWaker) -> bool {
        self.will_wake(&other.0)
    }
}

impl Clone for Waker {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            self.inner.as_ref().clone_raw()
        }
    }
}

impl fmt::Debug for Waker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Waker")
            .finish()
    }
}

impl Drop for Waker {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.inner.as_ref().drop_raw()
        }
    }
}

/// A `LocalWaker` is a handle for waking up a task by notifying its executor that it
/// is ready to be run.
///
/// This is similar to the `Waker` type, but cannot be sent across threads.
/// Task executors can use this type to implement more optimized single-threaded wakeup
/// behavior.
#[repr(transparent)]
#[derive(Clone)]
pub struct LocalWaker(Waker);

impl Unpin for LocalWaker {}
impl !Send for LocalWaker {}
impl !Sync for LocalWaker {}

impl LocalWaker {
    /// Constructs a new `LocalWaker` directly.
    ///
    /// Note that most code will not need to call this. Implementers of the
    /// `UnsafeWake` trait will typically provide a wrapper that calls this
    /// but you otherwise shouldn't call it directly.
    ///
    /// If you're working with the standard library then it's recommended to
    /// use the `local_waker_from_nonlocal` or `local_waker` to convert a `Waker`
    /// into a `LocalWaker`.
    ///
    /// For this function to be used safely, it must be sound to call `inner.wake_local()`
    /// on the current thread.
    #[inline]
    pub unsafe fn new(inner: NonNull<dyn UnsafeWake>) -> Self {
        LocalWaker(Waker::new(inner))
    }

    /// Borrows this `LocalWaker` as a `Waker`.
    ///
    /// `Waker` is nearly identical to `LocalWaker`, but is threadsafe
    /// (implements `Send` and `Sync`).
    #[inline]
    pub fn as_waker(&self) -> &Waker {
        &self.0
    }

    /// Converts this `LocalWaker` into a `Waker`.
    ///
    /// `Waker` is nearly identical to `LocalWaker`, but is threadsafe
    /// (implements `Send` and `Sync`).
    #[inline]
    pub fn into_waker(self) -> Waker {
        self.0
    }

    /// Wake up the task associated with this `LocalWaker`.
    #[inline]
    pub fn wake(&self) {
        unsafe { self.0.inner.as_ref().wake_local() }
    }

    /// Returns `true` if or not this `LocalWaker` and `other` `LocalWaker` awaken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `LocalWaker`s would awaken the same task. However, if this function
    /// returns true, it is guaranteed that the `LocalWaker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake(&self, other: &LocalWaker) -> bool {
        self.0.will_wake(&other.0)
    }

    /// Returns `true` if or not this `LocalWaker` and `other` `Waker` awaken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns true, it is guaranteed that the `LocalWaker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake_nonlocal(&self, other: &Waker) -> bool {
        self.0.will_wake(other)
    }
}

impl From<LocalWaker> for Waker {
    /// Converts a `LocalWaker` into a `Waker`.
    ///
    /// This conversion turns a `!Sync` `LocalWaker` into a `Sync` `Waker`, allowing a wakeup
    /// object to be sent to another thread, but giving up its ability to do specialized
    /// thread-local wakeup behavior.
    #[inline]
    fn from(local_waker: LocalWaker) -> Self {
        local_waker.0
    }
}

impl fmt::Debug for LocalWaker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LocalWaker")
            .finish()
    }
}

/// An unsafe trait for implementing custom memory management for a `Waker` or `LocalWaker`.
///
/// A `Waker` conceptually is a cloneable trait object for `Wake`, and is
/// most often essentially just `Arc<dyn Wake>`. However, in some contexts
/// (particularly `no_std`), it's desirable to avoid `Arc` in favor of some
/// custom memory management strategy. This trait is designed to allow for such
/// customization.
///
/// When using `std`, a default implementation of the `UnsafeWake` trait is provided for
/// `Arc<T>` where `T: Wake`.
pub unsafe trait UnsafeWake: Send + Sync {
    /// Creates a clone of this `UnsafeWake` and stores it behind a `Waker`.
    ///
    /// This function will create a new uniquely owned handle that under the
    /// hood references the same notification instance. In other words calls
    /// to `wake` on the returned handle should be equivalent to calls to
    /// `wake` on this handle.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe to call because it's asserting the `UnsafeWake`
    /// value is in a consistent state, i.e., hasn't been dropped.
    unsafe fn clone_raw(&self) -> Waker;

    /// Drops this instance of `UnsafeWake`, deallocating resources
    /// associated with it.
    ///
    // FIXME(cramertj):
    /// This method is intended to have a signature such as:
    ///
    /// ```ignore (not-a-doctest)
    /// fn drop_raw(self: *mut Self);
    /// ```
    ///
    /// Unfortunately, in Rust today that signature is not object safe.
    /// Nevertheless it's recommended to implement this function *as if* that
    /// were its signature. As such it is not safe to call on an invalid
    /// pointer, nor is the validity of the pointer guaranteed after this
    /// function returns.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe to call because it's asserting the `UnsafeWake`
    /// value is in a consistent state, i.e., hasn't been dropped.
    unsafe fn drop_raw(&self);

    /// Indicates that the associated task is ready to make progress and should
    /// be `poll`ed.
    ///
    /// Executors generally maintain a queue of "ready" tasks; `wake` should place
    /// the associated task onto this queue.
    ///
    /// # Panics
    ///
    /// Implementations should avoid panicking, but clients should also be prepared
    /// for panics.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe to call because it's asserting the `UnsafeWake`
    /// value is in a consistent state, i.e., hasn't been dropped.
    unsafe fn wake(&self);

    /// Indicates that the associated task is ready to make progress and should
    /// be `poll`ed. This function is the same as `wake`, but can only be called
    /// from the thread that this `UnsafeWake` is "local" to. This allows for
    /// implementors to provide specialized wakeup behavior specific to the current
    /// thread. This function is called by `LocalWaker::wake`.
    ///
    /// Executors generally maintain a queue of "ready" tasks; `wake_local` should place
    /// the associated task onto this queue.
    ///
    /// # Panics
    ///
    /// Implementations should avoid panicking, but clients should also be prepared
    /// for panics.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe to call because it's asserting the `UnsafeWake`
    /// value is in a consistent state, i.e., hasn't been dropped, and that the
    /// `UnsafeWake` hasn't moved from the thread on which it was created.
    unsafe fn wake_local(&self) {
        self.wake()
    }
}
