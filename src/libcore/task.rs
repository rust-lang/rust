// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

//! Types and Traits for working with asynchronous tasks.

use fmt;
use ptr::NonNull;
use future::Future;
use mem::PinMut;

/// Indicates whether a value is available or if the current task has been
/// scheduled to receive a wakeup instead.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Poll<T> {
    /// Represents that a value is immediately ready.
    Ready(T),

    /// Represents that a value is not ready yet.
    ///
    /// When a function returns `Pending`, the function *must* also
    /// ensure that the current task is scheduled to be awoken when
    /// progress can be made.
    Pending,
}

impl<T> Poll<T> {
    /// Change the ready value of this `Poll` with the closure provided
    pub fn map<U, F>(self, f: F) -> Poll<U>
        where F: FnOnce(T) -> U
    {
        match self {
            Poll::Ready(t) => Poll::Ready(f(t)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Returns whether this is `Poll::Ready`
    pub fn is_ready(&self) -> bool {
        match *self {
            Poll::Ready(_) => true,
            Poll::Pending => false,
        }
    }

    /// Returns whether this is `Poll::Pending`
    pub fn is_pending(&self) -> bool {
        !self.is_ready()
    }
}

impl<T, E> Poll<Result<T, E>> {
    /// Change the success value of this `Poll` with the closure provided
    pub fn map_ok<U, F>(self, f: F) -> Poll<Result<U, E>>
        where F: FnOnce(T) -> U
    {
        match self {
            Poll::Ready(Ok(t)) => Poll::Ready(Ok(f(t))),
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Change the error value of this `Poll` with the closure provided
    pub fn map_err<U, F>(self, f: F) -> Poll<Result<T, U>>
        where F: FnOnce(E) -> U
    {
        match self {
            Poll::Ready(Ok(t)) => Poll::Ready(Ok(t)),
            Poll::Ready(Err(e)) => Poll::Ready(Err(f(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> From<T> for Poll<T> {
    fn from(t: T) -> Poll<T> {
        Poll::Ready(t)
    }
}

/// A `Waker` is a handle for waking up a task by notifying its executor that it
/// is ready to be run.
///
/// This handle contains a trait object pointing to an instance of the `UnsafeWake`
/// trait, allowing notifications to get routed through it.
#[repr(transparent)]
pub struct Waker {
    inner: NonNull<UnsafeWake>,
}

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
    pub unsafe fn new(inner: NonNull<UnsafeWake>) -> Self {
        Waker { inner: inner }
    }

    /// Wake up the task associated with this `Waker`.
    #[inline]
    pub fn wake(&self) {
        unsafe { self.inner.as_ref().wake() }
    }

    /// Returns whether or not this `Waker` and `other` awaken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns true, it is guaranteed that the `Waker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake(&self, other: &Waker) -> bool {
        self.inner == other.inner
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
/// Task executors can use this type to implement more optimized singlethreaded wakeup
/// behavior.
#[repr(transparent)]
pub struct LocalWaker {
    inner: NonNull<UnsafeWake>,
}

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
    /// use the `LocalWaker::from` function instead which works with the safe
    /// `Rc` type and the safe `LocalWake` trait.
    ///
    /// For this function to be used safely, it must be sound to call `inner.wake_local()`
    /// on the current thread.
    #[inline]
    pub unsafe fn new(inner: NonNull<UnsafeWake>) -> Self {
        LocalWaker { inner: inner }
    }

    /// Wake up the task associated with this `LocalWaker`.
    #[inline]
    pub fn wake(&self) {
        unsafe { self.inner.as_ref().wake_local() }
    }

    /// Returns whether or not this `LocalWaker` and `other` `LocalWaker` awaken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `LocalWaker`s would awaken the same task. However, if this function
    /// returns true, it is guaranteed that the `LocalWaker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake(&self, other: &LocalWaker) -> bool {
        self.inner == other.inner
    }

    /// Returns whether or not this `LocalWaker` and `other` `Waker` awaken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns true, it is guaranteed that the `LocalWaker`s will awaken the same
    /// task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    pub fn will_wake_nonlocal(&self, other: &Waker) -> bool {
        self.inner == other.inner
    }
}

impl From<LocalWaker> for Waker {
    #[inline]
    fn from(local_waker: LocalWaker) -> Self {
        Waker { inner: local_waker.inner }
    }
}

impl Clone for LocalWaker {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            LocalWaker { inner: self.inner.as_ref().clone_raw().inner }
        }
    }
}

impl fmt::Debug for LocalWaker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Waker")
            .finish()
    }
}

impl Drop for LocalWaker {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.inner.as_ref().drop_raw()
        }
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
/// `Arc<T>` where `T: Wake` and `Rc<T>` where `T: LocalWake`.
///
/// Although the methods on `UnsafeWake` take pointers rather than references,
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
    /// value is in a consistent state, i.e. hasn't been dropped.
    unsafe fn clone_raw(&self) -> Waker;

    /// Drops this instance of `UnsafeWake`, deallocating resources
    /// associated with it.
    ///
    /// FIXME(cramertj)
    /// This method is intended to have a signature such as:
    ///
    /// ```ignore (not-a-doctest)
    /// fn drop_raw(self: *mut Self);
    /// ```
    ///
    /// Unfortunately in Rust today that signature is not object safe.
    /// Nevertheless it's recommended to implement this function *as if* that
    /// were its signature. As such it is not safe to call on an invalid
    /// pointer, nor is the validity of the pointer guaranteed after this
    /// function returns.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe to call because it's asserting the `UnsafeWake`
    /// value is in a consistent state, i.e. hasn't been dropped.
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
    /// value is in a consistent state, i.e. hasn't been dropped.
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
    /// value is in a consistent state, i.e. hasn't been dropped, and that the
    /// `UnsafeWake` hasn't moved from the thread on which it was created.
    unsafe fn wake_local(&self) {
        self.wake()
    }
}

/// Information about the currently-running task.
///
/// Contexts are always tied to the stack, since they are set up specifically
/// when performing a single `poll` step on a task.
pub struct Context<'a> {
    local_waker: &'a LocalWaker,
    executor: &'a mut Executor,
}

impl<'a> fmt::Debug for Context<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Context")
            .finish()
    }
}

impl<'a> Context<'a> {
    /// Create a new task `Context` with the provided `local_waker`, `waker`, and `executor`.
    #[inline]
    pub fn new(local_waker: &'a LocalWaker, executor: &'a mut Executor) -> Context<'a> {
        Context {
            local_waker,
            executor,
        }
    }

    /// Get the `LocalWaker` associated with the current task.
    #[inline]
    pub fn local_waker(&self) -> &'a LocalWaker {
        self.local_waker
    }

    /// Get the `Waker` associated with the current task.
    #[inline]
    pub fn waker(&self) -> &'a Waker {
        unsafe { &*(self.local_waker as *const LocalWaker as *const Waker) }
    }

    /// Get the default executor associated with this task.
    ///
    /// This method is useful primarily if you want to explicitly handle
    /// spawn failures.
    #[inline]
    pub fn executor(&mut self) -> &mut Executor {
        self.executor
    }

    /// Produce a context like the current one, but using the given waker instead.
    ///
    /// This advanced method is primarily used when building "internal
    /// schedulers" within a task, where you want to provide some customized
    /// wakeup logic.
    #[inline]
    pub fn with_waker<'b>(&'b mut self, local_waker: &'b LocalWaker) -> Context<'b> {
        Context {
            local_waker,
            executor: self.executor,
        }
    }

    /// Produce a context like the current one, but using the given executor
    /// instead.
    ///
    /// This advanced method is primarily used when building "internal
    /// schedulers" within a task.
    #[inline]
    pub fn with_executor<'b, E>(&'b mut self, executor: &'b mut E) -> Context<'b>
        where E: Executor
    {
        Context {
            local_waker: self.local_waker,
            executor: executor,
        }
    }
}

/// A task executor.
///
/// A *task* is a `()`-producing async value that runs at the top level, and will
/// be `poll`ed until completion. It's also the unit at which wake-up
/// notifications occur. Executors, such as thread pools, allow tasks to be
/// spawned and are responsible for putting tasks onto ready queues when
/// they are woken up, and polling them when they are ready.
pub trait Executor {
    /// Spawn the given task, polling it until completion.
    ///
    /// # Errors
    ///
    /// The executor may be unable to spawn tasks, either because it has
    /// been shut down or is resource-constrained.
    fn spawn_obj(&mut self, task: TaskObj) -> Result<(), SpawnObjError>;

    /// Determine whether the executor is able to spawn new tasks.
    ///
    /// # Returns
    ///
    /// An `Ok` return means the executor is *likely* (but not guaranteed)
    /// to accept a subsequent spawn attempt. Likewise, an `Err` return
    /// means that `spawn` is likely, but not guaranteed, to yield an error.
    #[inline]
    fn status(&self) -> Result<(), SpawnErrorKind> {
        Ok(())
    }
}

/// A custom trait object for polling tasks, roughly akin to
/// `Box<Future<Output = ()> + Send>`.
pub struct TaskObj {
    ptr: *mut (),
    poll_fn: unsafe fn(*mut (), &mut Context) -> Poll<()>,
    drop_fn: unsafe fn(*mut ()),
}

impl fmt::Debug for TaskObj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TaskObj")
            .finish()
    }
}

unsafe impl Send for TaskObj {}

/// A custom implementation of a task trait object for `TaskObj`, providing
/// a hand-rolled vtable.
///
/// This custom representation is typically used only in `no_std` contexts,
/// where the default `Box`-based implementation is not available.
///
/// The implementor must guarantee that it is safe to call `poll` repeatedly (in
/// a non-concurrent fashion) with the result of `into_raw` until `drop` is
/// called.
pub unsafe trait UnsafeTask: Send + 'static {
    /// Convert a owned instance into a (conceptually owned) void pointer.
    fn into_raw(self) -> *mut ();

    /// Poll the task represented by the given void pointer.
    ///
    /// # Safety
    ///
    /// The trait implementor must guarantee that it is safe to repeatedly call
    /// `poll` with the result of `into_raw` until `drop` is called; such calls
    /// are not, however, allowed to race with each other or with calls to `drop`.
    unsafe fn poll(task: *mut (), cx: &mut Context) -> Poll<()>;

    /// Drops the task represented by the given void pointer.
    ///
    /// # Safety
    ///
    /// The trait implementor must guarantee that it is safe to call this
    /// function once per `into_raw` invocation; that call cannot race with
    /// other calls to `drop` or `poll`.
    unsafe fn drop(task: *mut ());
}

impl TaskObj {
    /// Create a `TaskObj` from a custom trait object representation.
    #[inline]
    pub fn new<T: UnsafeTask>(t: T) -> TaskObj {
        TaskObj {
            ptr: t.into_raw(),
            poll_fn: T::poll,
            drop_fn: T::drop,
        }
    }
}

impl Future for TaskObj {
    type Output = ();

    #[inline]
    fn poll(self: PinMut<Self>, cx: &mut Context) -> Poll<()> {
        unsafe {
            (self.poll_fn)(self.ptr, cx)
        }
    }
}

impl Drop for TaskObj {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self.ptr)
        }
    }
}

/// Provides the reason that an executor was unable to spawn.
pub struct SpawnErrorKind {
    _hidden: (),
}

impl fmt::Debug for SpawnErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("SpawnErrorKind")
            .field(&"shutdown")
            .finish()
    }
}

impl SpawnErrorKind {
    /// Spawning is failing because the executor has been shut down.
    pub fn shutdown() -> SpawnErrorKind {
        SpawnErrorKind { _hidden: () }
    }

    /// Check whether this error is the `shutdown` error.
    pub fn is_shutdown(&self) -> bool {
        true
    }
}

/// The result of a failed spawn
#[derive(Debug)]
pub struct SpawnObjError {
    /// The kind of error
    pub kind: SpawnErrorKind,

    /// The task for which spawning was attempted
    pub task: TaskObj,
}
