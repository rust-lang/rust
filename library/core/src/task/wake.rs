#![stable(feature = "futures_api", since = "1.36.0")]

use crate::any::Any;
use crate::marker::PhantomData;
use crate::mem::{ManuallyDrop, transmute};
use crate::panic::AssertUnwindSafe;
use crate::{fmt, ptr};

/// A `RawWaker` allows the implementor of a task executor to create a [`Waker`]
/// or a [`LocalWaker`] which provides customized wakeup behavior.
///
/// It consists of a data pointer and a [virtual function pointer table (vtable)][vtable]
/// that customizes the behavior of the `RawWaker`.
///
/// `RawWaker`s are unsafe to use.
/// Implementing the [`Wake`] trait is a safe alternative that requires memory allocation.
///
/// [vtable]: https://en.wikipedia.org/wiki/Virtual_method_table
/// [`Wake`]: ../../alloc/task/trait.Wake.html
#[derive(PartialEq, Debug)]
#[stable(feature = "futures_api", since = "1.36.0")]
pub struct RawWaker {
    /// A data pointer, which can be used to store arbitrary data as required
    /// by the executor. This could be e.g. a type-erased pointer to an `Arc`
    /// that is associated with the task.
    /// The value of this field gets passed to all functions that are part of
    /// the vtable as the first parameter.
    data: *const (),
    /// Virtual function pointer table that customizes the behavior of this waker.
    vtable: &'static RawWakerVTable,
}

impl RawWaker {
    /// Creates a new `RawWaker` from the provided `data` pointer and `vtable`.
    ///
    /// The `data` pointer can be used to store arbitrary data as required
    /// by the executor. This could be e.g. a type-erased pointer to an `Arc`
    /// that is associated with the task.
    /// The value of this pointer will get passed to all functions that are part
    /// of the `vtable` as the first parameter.
    ///
    /// It is important to consider that the `data` pointer must point to a
    /// thread safe type such as an `Arc<T: Send + Sync>`
    /// when used to construct a [`Waker`]. This restriction is lifted when
    /// constructing a [`LocalWaker`], which allows using types that do not implement
    /// <code>[Send] + [Sync]</code> like `Rc<T>`.
    ///
    /// The `vtable` customizes the behavior of a `Waker` which gets created
    /// from a `RawWaker`. For each operation on the `Waker`, the associated
    /// function in the `vtable` of the underlying `RawWaker` will be called.
    #[inline]
    #[rustc_promotable]
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "futures_api", since = "1.36.0")]
    #[must_use]
    pub const fn new(data: *const (), vtable: &'static RawWakerVTable) -> RawWaker {
        RawWaker { data, vtable }
    }

    #[stable(feature = "noop_waker", since = "1.85.0")]
    const NOOP: RawWaker = {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            // Cloning just returns a new no-op raw waker
            |_| RawWaker::NOOP,
            // `wake` does nothing
            |_| {},
            // `wake_by_ref` does nothing
            |_| {},
            // Dropping does nothing as we don't allocate anything
            |_| {},
        );
        RawWaker::new(ptr::null(), &VTABLE)
    };
}

/// A virtual function pointer table (vtable) that specifies the behavior
/// of a [`RawWaker`].
///
/// The pointer passed to all functions inside the vtable is the `data` pointer
/// from the enclosing [`RawWaker`] object.
///
/// The functions inside this struct are only intended to be called on the `data`
/// pointer of a properly constructed [`RawWaker`] object from inside the
/// [`RawWaker`] implementation. Calling one of the contained functions using
/// any other `data` pointer will cause undefined behavior.
///
/// Note that while this type implements `PartialEq`, comparing function pointers, and hence
/// comparing structs like this that contain function pointers, is unreliable: pointers to the same
/// function can compare inequal (because functions are duplicated in multiple codegen units), and
/// pointers to *different* functions can compare equal (since identical functions can be
/// deduplicated within a codegen unit).
///
/// # Thread safety
/// If the [`RawWaker`] will be used to construct a [`Waker`] then
/// these functions must all be thread-safe (even though [`RawWaker`] is
/// <code>\![Send] + \![Sync]</code>). This is because [`Waker`] is <code>[Send] + [Sync]</code>,
/// and it may be moved to arbitrary threads or invoked by `&` reference. For example,
/// this means that if the `clone` and `drop` functions manage a reference count,
/// they must do so atomically.
///
/// However, if the [`RawWaker`] will be used to construct a [`LocalWaker`] instead, then
/// these functions don't need to be thread safe. This means that <code>\![Send] + \![Sync]</code>
///  data can be stored in the data pointer, and reference counting does not need any atomic
/// synchronization. This is because [`LocalWaker`] is not thread safe itself, so it cannot
/// be sent across threads.
#[stable(feature = "futures_api", since = "1.36.0")]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct RawWakerVTable {
    /// This function will be called when the [`RawWaker`] gets cloned, e.g. when
    /// the [`Waker`] in which the [`RawWaker`] is stored gets cloned.
    ///
    /// The implementation of this function must retain all resources that are
    /// required for this additional instance of a [`RawWaker`] and associated
    /// task. Calling `wake` on the resulting [`RawWaker`] should result in a wakeup
    /// of the same task that would have been awoken by the original [`RawWaker`].
    clone: unsafe fn(*const ()) -> RawWaker,

    /// This function will be called when `wake` is called on the [`Waker`].
    /// It must wake up the task associated with this [`RawWaker`].
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    wake: unsafe fn(*const ()),

    /// This function will be called when `wake_by_ref` is called on the [`Waker`].
    /// It must wake up the task associated with this [`RawWaker`].
    ///
    /// This function is similar to `wake`, but must not consume the provided data
    /// pointer.
    wake_by_ref: unsafe fn(*const ()),

    /// This function will be called when a [`Waker`] gets dropped.
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    drop: unsafe fn(*const ()),
}

impl RawWakerVTable {
    /// Creates a new `RawWakerVTable` from the provided `clone`, `wake`,
    /// `wake_by_ref`, and `drop` functions.
    ///
    /// If the [`RawWaker`] will be used to construct a [`Waker`] then
    /// these functions must all be thread-safe (even though [`RawWaker`] is
    /// <code>\![Send] + \![Sync]</code>). This is because [`Waker`] is <code>[Send] + [Sync]</code>,
    /// and it may be moved to arbitrary threads or invoked by `&` reference. For example,
    /// this means that if the `clone` and `drop` functions manage a reference count,
    /// they must do so atomically.
    ///
    /// However, if the [`RawWaker`] will be used to construct a [`LocalWaker`] instead, then
    /// these functions don't need to be thread safe. This means that <code>\![Send] + \![Sync]</code>
    /// data can be stored in the data pointer, and reference counting does not need any atomic
    /// synchronization. This is because [`LocalWaker`] is not thread safe itself, so it cannot
    /// be sent across threads.
    /// # `clone`
    ///
    /// This function will be called when the [`RawWaker`] gets cloned, e.g. when
    /// the [`Waker`]/[`LocalWaker`] in which the [`RawWaker`] is stored gets cloned.
    ///
    /// The implementation of this function must retain all resources that are
    /// required for this additional instance of a [`RawWaker`] and associated
    /// task. Calling `wake` on the resulting [`RawWaker`] should result in a wakeup
    /// of the same task that would have been awoken by the original [`RawWaker`].
    ///
    /// # `wake`
    ///
    /// This function will be called when `wake` is called on the [`Waker`].
    /// It must wake up the task associated with this [`RawWaker`].
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    ///
    /// # `wake_by_ref`
    ///
    /// This function will be called when `wake_by_ref` is called on the [`Waker`].
    /// It must wake up the task associated with this [`RawWaker`].
    ///
    /// This function is similar to `wake`, but must not consume the provided data
    /// pointer.
    ///
    /// # `drop`
    ///
    /// This function will be called when a [`Waker`]/[`LocalWaker`] gets
    /// dropped.
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    #[rustc_promotable]
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "futures_api", since = "1.36.0")]
    pub const fn new(
        clone: unsafe fn(*const ()) -> RawWaker,
        wake: unsafe fn(*const ()),
        wake_by_ref: unsafe fn(*const ()),
        drop: unsafe fn(*const ()),
    ) -> Self {
        Self { clone, wake, wake_by_ref, drop }
    }
}

#[derive(Debug)]
enum ExtData<'a> {
    Some(&'a mut dyn Any),
    None(()),
}

/// The context of an asynchronous task.
///
/// Currently, `Context` only serves to provide access to a [`&Waker`](Waker)
/// which can be used to wake the current task.
#[stable(feature = "futures_api", since = "1.36.0")]
#[lang = "Context"]
pub struct Context<'a> {
    waker: &'a Waker,
    local_waker: &'a LocalWaker,
    ext: AssertUnwindSafe<ExtData<'a>>,
    // Ensure we future-proof against variance changes by forcing
    // the lifetime to be invariant (argument-position lifetimes
    // are contravariant while return-position lifetimes are
    // covariant).
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
    // Ensure `Context` is `!Send` and `!Sync` in order to allow
    // for future `!Send` and / or `!Sync` fields.
    _marker2: PhantomData<*mut ()>,
}

impl<'a> Context<'a> {
    /// Creates a new `Context` from a [`&Waker`](Waker).
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "const_waker", since = "1.82.0")]
    #[must_use]
    #[inline]
    pub const fn from_waker(waker: &'a Waker) -> Self {
        ContextBuilder::from_waker(waker).build()
    }

    /// Returns a reference to the [`Waker`] for the current task.
    #[inline]
    #[must_use]
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "const_waker", since = "1.82.0")]
    pub const fn waker(&self) -> &'a Waker {
        &self.waker
    }

    /// Returns a reference to the [`LocalWaker`] for the current task.
    #[inline]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const fn local_waker(&self) -> &'a LocalWaker {
        &self.local_waker
    }

    /// Returns a reference to the extension data for the current task.
    #[inline]
    #[unstable(feature = "context_ext", issue = "123392")]
    pub const fn ext(&mut self) -> &mut dyn Any {
        // FIXME: this field makes Context extra-weird about unwind safety
        // can we justify AssertUnwindSafe if we stabilize this? do we care?
        match &mut self.ext.0 {
            ExtData::Some(data) => *data,
            ExtData::None(unit) => unit,
        }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl fmt::Debug for Context<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").field("waker", &self.waker).finish()
    }
}

/// A Builder used to construct a `Context` instance
/// with support for `LocalWaker`.
///
/// # Examples
/// ```
/// #![feature(local_waker)]
/// use std::task::{ContextBuilder, LocalWaker, Waker, Poll};
/// use std::future::Future;
///
/// let local_waker = LocalWaker::noop();
/// let waker = Waker::noop();
///
/// let mut cx = ContextBuilder::from_waker(&waker)
///     .local_waker(&local_waker)
///     .build();
///
/// let mut future = std::pin::pin!(async { 20 });
/// let poll = future.as_mut().poll(&mut cx);
/// assert_eq!(poll, Poll::Ready(20));
///
/// ```
#[unstable(feature = "local_waker", issue = "118959")]
#[derive(Debug)]
pub struct ContextBuilder<'a> {
    waker: &'a Waker,
    local_waker: &'a LocalWaker,
    ext: ExtData<'a>,
    // Ensure we future-proof against variance changes by forcing
    // the lifetime to be invariant (argument-position lifetimes
    // are contravariant while return-position lifetimes are
    // covariant).
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
    // Ensure `Context` is `!Send` and `!Sync` in order to allow
    // for future `!Send` and / or `!Sync` fields.
    _marker2: PhantomData<*mut ()>,
}

impl<'a> ContextBuilder<'a> {
    /// Creates a ContextBuilder from a Waker.
    #[inline]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const fn from_waker(waker: &'a Waker) -> Self {
        // SAFETY: LocalWaker is just Waker without thread safety
        let local_waker = unsafe { transmute(waker) };
        Self {
            waker,
            local_waker,
            ext: ExtData::None(()),
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    /// Creates a ContextBuilder from an existing Context.
    #[inline]
    #[unstable(feature = "context_ext", issue = "123392")]
    pub const fn from(cx: &'a mut Context<'_>) -> Self {
        let ext = match &mut cx.ext.0 {
            ExtData::Some(ext) => ExtData::Some(*ext),
            ExtData::None(()) => ExtData::None(()),
        };
        Self {
            waker: cx.waker,
            local_waker: cx.local_waker,
            ext,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    /// Sets the value for the waker on `Context`.
    #[inline]
    #[unstable(feature = "context_ext", issue = "123392")]
    pub const fn waker(self, waker: &'a Waker) -> Self {
        Self { waker, ..self }
    }

    /// Sets the value for the local waker on `Context`.
    #[inline]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const fn local_waker(self, local_waker: &'a LocalWaker) -> Self {
        Self { local_waker, ..self }
    }

    /// Sets the value for the extension data on `Context`.
    #[inline]
    #[unstable(feature = "context_ext", issue = "123392")]
    pub const fn ext(self, data: &'a mut dyn Any) -> Self {
        Self { ext: ExtData::Some(data), ..self }
    }

    /// Builds the `Context`.
    #[inline]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const fn build(self) -> Context<'a> {
        let ContextBuilder { waker, local_waker, ext, _marker, _marker2 } = self;
        Context { waker, local_waker, ext: AssertUnwindSafe(ext), _marker, _marker2 }
    }
}

/// A `Waker` is a handle for waking up a task by notifying its executor that it
/// is ready to be run.
///
/// This handle encapsulates a [`RawWaker`] instance, which defines the
/// executor-specific wakeup behavior.
///
/// The typical life of a `Waker` is that it is constructed by an executor, wrapped in a
/// [`Context`], then passed to [`Future::poll()`]. Then, if the future chooses to return
/// [`Poll::Pending`], it must also store the waker somehow and call [`Waker::wake()`] when
/// the future should be polled again.
///
/// Implements [`Clone`], [`Send`], and [`Sync`]; therefore, a waker may be invoked
/// from any thread, including ones not in any way managed by the executor. For example,
/// this might be done to wake a future when a blocking function call completes on another
/// thread.
///
/// Note that it is preferable to use `waker.clone_from(&new_waker)` instead
/// of `*waker = new_waker.clone()`, as the former will avoid cloning the waker
/// unnecessarily if the two wakers [wake the same task](Self::will_wake).
///
/// Constructing a `Waker` from a [`RawWaker`] is unsafe.
/// Implementing the [`Wake`] trait is a safe alternative that requires memory allocation.
///
/// [`Future::poll()`]: core::future::Future::poll
/// [`Poll::Pending`]: core::task::Poll::Pending
/// [`Wake`]: ../../alloc/task/trait.Wake.html
#[repr(transparent)]
#[stable(feature = "futures_api", since = "1.36.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Waker")]
pub struct Waker {
    waker: RawWaker,
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl Unpin for Waker {}
#[stable(feature = "futures_api", since = "1.36.0")]
unsafe impl Send for Waker {}
#[stable(feature = "futures_api", since = "1.36.0")]
unsafe impl Sync for Waker {}

impl Waker {
    /// Wakes up the task associated with this `Waker`.
    ///
    /// As long as the executor keeps running and the task is not finished, it is
    /// guaranteed that each invocation of [`wake()`](Self::wake) (or
    /// [`wake_by_ref()`](Self::wake_by_ref)) will be followed by at least one
    /// [`poll()`] of the task to which this `Waker` belongs. This makes
    /// it possible to temporarily yield to other tasks while running potentially
    /// unbounded processing loops.
    ///
    /// Note that the above implies that multiple wake-ups may be coalesced into a
    /// single [`poll()`] invocation by the runtime.
    ///
    /// Also note that yielding to competing tasks is not guaranteed: it is the
    /// executor’s choice which task to run and the executor may choose to run the
    /// current task again.
    ///
    /// [`poll()`]: crate::future::Future::poll
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn wake(self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.

        // Don't call `drop` -- the waker will be consumed by `wake`.
        let this = ManuallyDrop::new(self);

        // SAFETY: This is safe because `Waker::from_raw` is the only way
        // to initialize `wake` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (this.waker.vtable.wake)(this.waker.data) };
    }

    /// Wakes up the task associated with this `Waker` without consuming the `Waker`.
    ///
    /// This is similar to [`wake()`](Self::wake), but may be slightly less efficient in
    /// the case where an owned `Waker` is available. This method should be preferred to
    /// calling `waker.clone().wake()`.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn wake_by_ref(&self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.

        // SAFETY: see `wake`
        unsafe { (self.waker.vtable.wake_by_ref)(self.waker.data) }
    }

    /// Returns `true` if this `Waker` and another `Waker` would awake the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns `true`, it is guaranteed that the `Waker`s will awaken the same task.
    ///
    /// This function is primarily used for optimization purposes — for example,
    /// this type's [`clone_from`](Self::clone_from) implementation uses it to
    /// avoid cloning the waker when they would wake the same task anyway.
    #[inline]
    #[must_use]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn will_wake(&self, other: &Waker) -> bool {
        // We optimize this by comparing vtable addresses instead of vtable contents.
        // This is permitted since the function is documented as best-effort.
        let RawWaker { data: a_data, vtable: a_vtable } = self.waker;
        let RawWaker { data: b_data, vtable: b_vtable } = other.waker;
        a_data == b_data && ptr::eq(a_vtable, b_vtable)
    }

    /// Creates a new `Waker` from the provided `data` pointer and `vtable`.
    ///
    /// The `data` pointer can be used to store arbitrary data as required
    /// by the executor. This could be e.g. a type-erased pointer to an `Arc`
    /// that is associated with the task.
    /// The value of this pointer will get passed to all functions that are part
    /// of the `vtable` as the first parameter.
    ///
    /// It is important to consider that the `data` pointer must point to a
    /// thread safe type such as an `Arc`.
    ///
    /// The `vtable` customizes the behavior of a `Waker`. For each operation
    /// on the `Waker`, the associated function in the `vtable` will be called.
    ///
    /// # Safety
    ///
    /// The behavior of the returned `Waker` is undefined if the contract defined
    /// in [`RawWakerVTable`]'s documentation is not upheld.
    ///
    /// (Authors wishing to avoid unsafe code may implement the [`Wake`] trait instead, at the
    /// cost of a required heap allocation.)
    ///
    /// [`Wake`]: ../../alloc/task/trait.Wake.html
    #[inline]
    #[must_use]
    #[stable(feature = "waker_getters", since = "1.83.0")]
    #[rustc_const_stable(feature = "waker_getters", since = "1.83.0")]
    pub const unsafe fn new(data: *const (), vtable: &'static RawWakerVTable) -> Self {
        Waker { waker: RawWaker { data, vtable } }
    }

    /// Creates a new `Waker` from [`RawWaker`].
    ///
    /// # Safety
    ///
    /// The behavior of the returned `Waker` is undefined if the contract defined
    /// in [`RawWaker`]'s and [`RawWakerVTable`]'s documentation is not upheld.
    ///
    /// (Authors wishing to avoid unsafe code may implement the [`Wake`] trait instead, at the
    /// cost of a required heap allocation.)
    ///
    /// [`Wake`]: ../../alloc/task/trait.Wake.html
    #[inline]
    #[must_use]
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "const_waker", since = "1.82.0")]
    pub const unsafe fn from_raw(waker: RawWaker) -> Waker {
        Waker { waker }
    }

    /// Returns a reference to a `Waker` that does nothing when used.
    ///
    // Note!  Much of the documentation for this method is duplicated
    // in the docs for `LocalWaker::noop`.
    // If you edit it, consider editing the other copy too.
    //
    /// This is mostly useful for writing tests that need a [`Context`] to poll
    /// some futures, but are not expecting those futures to wake the waker or
    /// do not need to do anything specific if it happens.
    ///
    /// More generally, using `Waker::noop()` to poll a future
    /// means discarding the notification of when the future should be polled again.
    /// So it should only be used when such a notification will not be needed to make progress.
    ///
    /// If an owned `Waker` is needed, `clone()` this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::future::Future;
    /// use std::task;
    ///
    /// let mut cx = task::Context::from_waker(task::Waker::noop());
    ///
    /// let mut future = Box::pin(async { 10 });
    /// assert_eq!(future.as_mut().poll(&mut cx), task::Poll::Ready(10));
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "noop_waker", since = "1.85.0")]
    #[rustc_const_stable(feature = "noop_waker", since = "1.85.0")]
    pub const fn noop() -> &'static Waker {
        const WAKER: &Waker = &Waker { waker: RawWaker::NOOP };
        WAKER
    }

    /// Gets the `data` pointer used to create this `Waker`.
    #[inline]
    #[must_use]
    #[stable(feature = "waker_getters", since = "1.83.0")]
    pub fn data(&self) -> *const () {
        self.waker.data
    }

    /// Gets the `vtable` pointer used to create this `Waker`.
    #[inline]
    #[must_use]
    #[stable(feature = "waker_getters", since = "1.83.0")]
    pub fn vtable(&self) -> &'static RawWakerVTable {
        self.waker.vtable
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl Clone for Waker {
    #[inline]
    fn clone(&self) -> Self {
        Waker {
            // SAFETY: This is safe because `Waker::from_raw` is the only way
            // to initialize `clone` and `data` requiring the user to acknowledge
            // that the contract of [`RawWaker`] is upheld.
            waker: unsafe { (self.waker.vtable.clone)(self.waker.data) },
        }
    }

    /// Assigns a clone of `source` to `self`, unless [`self.will_wake(source)`][Waker::will_wake] anyway.
    ///
    /// This method is preferred over simply assigning `source.clone()` to `self`,
    /// as it avoids cloning the waker if `self` is already the same waker.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::future::Future;
    /// use std::pin::Pin;
    /// use std::sync::{Arc, Mutex};
    /// use std::task::{Context, Poll, Waker};
    ///
    /// struct Waiter {
    ///     shared: Arc<Mutex<Shared>>,
    /// }
    ///
    /// struct Shared {
    ///     waker: Waker,
    ///     // ...
    /// }
    ///
    /// impl Future for Waiter {
    ///     type Output = ();
    ///     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
    ///         let mut shared = self.shared.lock().unwrap();
    ///
    ///         // update the waker
    ///         shared.waker.clone_from(cx.waker());
    ///
    ///         // readiness logic ...
    /// #       Poll::Ready(())
    ///     }
    /// }
    ///
    /// ```
    #[inline]
    fn clone_from(&mut self, source: &Self) {
        if !self.will_wake(source) {
            *self = source.clone();
        }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl Drop for Waker {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe because `Waker::from_raw` is the only way
        // to initialize `drop` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (self.waker.vtable.drop)(self.waker.data) }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl fmt::Debug for Waker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vtable_ptr = self.waker.vtable as *const RawWakerVTable;
        f.debug_struct("Waker")
            .field("data", &self.waker.data)
            .field("vtable", &vtable_ptr)
            .finish()
    }
}

/// A `LocalWaker` is analogous to a [`Waker`], but it does not implement [`Send`] or [`Sync`].
///
/// This handle encapsulates a [`RawWaker`] instance, which defines the
/// executor-specific wakeup behavior.
///
/// Local wakers can be requested from a `Context` with the [`local_waker`] method.
///
/// The typical life of a `LocalWaker` is that it is constructed by an executor, wrapped in a
/// [`Context`] using [`ContextBuilder`], then passed to [`Future::poll()`]. Then, if the future chooses to return
/// [`Poll::Pending`], it must also store the waker somehow and call [`LocalWaker::wake()`] when
/// the future should be polled again.
///
/// Implements [`Clone`], but neither [`Send`] nor [`Sync`]; therefore, a local waker may
/// not be moved to other threads. In general, when deciding to use wakers or local wakers,
/// local wakers are preferable unless the waker needs to be sent across threads. This is because
/// wakers can incur in additional cost related to memory synchronization.
///
/// Note that it is preferable to use `local_waker.clone_from(&new_waker)` instead
/// of `*local_waker = new_waker.clone()`, as the former will avoid cloning the waker
/// unnecessarily if the two wakers [wake the same task](Self::will_wake).
///
/// # Examples
/// Usage of a local waker to implement a future analogous to `std::thread::yield_now()`.
/// ```
/// #![feature(local_waker)]
/// use std::future::{Future, poll_fn};
/// use std::task::Poll;
///
/// // a future that returns pending once.
/// fn yield_now() -> impl Future<Output=()> + Unpin {
///     let mut yielded = false;
///     poll_fn(move |cx| {
///         if !yielded {
///             yielded = true;
///             cx.local_waker().wake_by_ref();
///             return Poll::Pending;
///         }
///         return Poll::Ready(())
///     })
/// }
///
/// # async fn __() {
/// yield_now().await;
/// # }
/// ```
///
/// [`Future::poll()`]: core::future::Future::poll
/// [`Poll::Pending`]: core::task::Poll::Pending
/// [`local_waker`]: core::task::Context::local_waker
#[unstable(feature = "local_waker", issue = "118959")]
#[repr(transparent)]
pub struct LocalWaker {
    waker: RawWaker,
}

#[unstable(feature = "local_waker", issue = "118959")]
impl Unpin for LocalWaker {}

impl LocalWaker {
    /// Wakes up the task associated with this `LocalWaker`.
    ///
    /// As long as the executor keeps running and the task is not finished, it is
    /// guaranteed that each invocation of [`wake()`](Self::wake) (or
    /// [`wake_by_ref()`](Self::wake_by_ref)) will be followed by at least one
    /// [`poll()`] of the task to which this `LocalWaker` belongs. This makes
    /// it possible to temporarily yield to other tasks while running potentially
    /// unbounded processing loops.
    ///
    /// Note that the above implies that multiple wake-ups may be coalesced into a
    /// single [`poll()`] invocation by the runtime.
    ///
    /// Also note that yielding to competing tasks is not guaranteed: it is the
    /// executor’s choice which task to run and the executor may choose to run the
    /// current task again.
    ///
    /// [`poll()`]: crate::future::Future::poll
    #[inline]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub fn wake(self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.

        // Don't call `drop` -- the waker will be consumed by `wake`.
        let this = ManuallyDrop::new(self);

        // SAFETY: This is safe because `Waker::from_raw` is the only way
        // to initialize `wake` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (this.waker.vtable.wake)(this.waker.data) };
    }

    /// Wakes up the task associated with this `LocalWaker` without consuming the `LocalWaker`.
    ///
    /// This is similar to [`wake()`](Self::wake), but may be slightly less efficient in
    /// the case where an owned `Waker` is available. This method should be preferred to
    /// calling `waker.clone().wake()`.
    #[inline]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub fn wake_by_ref(&self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.

        // SAFETY: see `wake`
        unsafe { (self.waker.vtable.wake_by_ref)(self.waker.data) }
    }

    /// Returns `true` if this `LocalWaker` and another `LocalWaker` would awake the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns `true`, it is guaranteed that the `Waker`s will awaken the same task.
    ///
    /// This function is primarily used for optimization purposes — for example,
    /// this type's [`clone_from`](Self::clone_from) implementation uses it to
    /// avoid cloning the waker when they would wake the same task anyway.
    #[inline]
    #[must_use]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub fn will_wake(&self, other: &LocalWaker) -> bool {
        // We optimize this by comparing vtable addresses instead of vtable contents.
        // This is permitted since the function is documented as best-effort.
        let RawWaker { data: a_data, vtable: a_vtable } = self.waker;
        let RawWaker { data: b_data, vtable: b_vtable } = other.waker;
        a_data == b_data && ptr::eq(a_vtable, b_vtable)
    }

    /// Creates a new `LocalWaker` from the provided `data` pointer and `vtable`.
    ///
    /// The `data` pointer can be used to store arbitrary data as required
    /// by the executor. This could be e.g. a type-erased pointer to an `Arc`
    /// that is associated with the task.
    /// The value of this pointer will get passed to all functions that are part
    /// of the `vtable` as the first parameter.
    ///
    /// The `vtable` customizes the behavior of a `LocalWaker`. For each
    /// operation on the `LocalWaker`, the associated function in the `vtable`
    /// will be called.
    ///
    /// # Safety
    ///
    /// The behavior of the returned `Waker` is undefined if the contract defined
    /// in [`RawWakerVTable`]'s documentation is not upheld.
    ///
    #[inline]
    #[must_use]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const unsafe fn new(data: *const (), vtable: &'static RawWakerVTable) -> Self {
        LocalWaker { waker: RawWaker { data, vtable } }
    }

    /// Creates a new `LocalWaker` from [`RawWaker`].
    ///
    /// The behavior of the returned `LocalWaker` is undefined if the contract defined
    /// in [`RawWaker`]'s and [`RawWakerVTable`]'s documentation is not upheld.
    /// Therefore this method is unsafe.
    #[inline]
    #[must_use]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const unsafe fn from_raw(waker: RawWaker) -> LocalWaker {
        Self { waker }
    }

    /// Returns a reference to a `LocalWaker` that does nothing when used.
    ///
    // Note!  Much of the documentation for this method is duplicated
    // in the docs for `Waker::noop`.
    // If you edit it, consider editing the other copy too.
    //
    /// This is mostly useful for writing tests that need a [`Context`] to poll
    /// some futures, but are not expecting those futures to wake the waker or
    /// do not need to do anything specific if it happens.
    ///
    /// More generally, using `LocalWaker::noop()` to poll a future
    /// means discarding the notification of when the future should be polled again,
    /// So it should only be used when such a notification will not be needed to make progress.
    ///
    /// If an owned `LocalWaker` is needed, `clone()` this one.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_waker)]
    /// use std::future::Future;
    /// use std::task::{ContextBuilder, LocalWaker, Waker, Poll};
    ///
    /// let mut cx = ContextBuilder::from_waker(Waker::noop())
    ///     .local_waker(LocalWaker::noop())
    ///     .build();
    ///
    /// let mut future = Box::pin(async { 10 });
    /// assert_eq!(future.as_mut().poll(&mut cx), Poll::Ready(10));
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub const fn noop() -> &'static LocalWaker {
        const WAKER: &LocalWaker = &LocalWaker { waker: RawWaker::NOOP };
        WAKER
    }

    /// Gets the `data` pointer used to create this `LocalWaker`.
    #[inline]
    #[must_use]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub fn data(&self) -> *const () {
        self.waker.data
    }

    /// Gets the `vtable` pointer used to create this `LocalWaker`.
    #[inline]
    #[must_use]
    #[unstable(feature = "local_waker", issue = "118959")]
    pub fn vtable(&self) -> &'static RawWakerVTable {
        self.waker.vtable
    }
}
#[unstable(feature = "local_waker", issue = "118959")]
impl Clone for LocalWaker {
    #[inline]
    fn clone(&self) -> Self {
        LocalWaker {
            // SAFETY: This is safe because `Waker::from_raw` is the only way
            // to initialize `clone` and `data` requiring the user to acknowledge
            // that the contract of [`RawWaker`] is upheld.
            waker: unsafe { (self.waker.vtable.clone)(self.waker.data) },
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        if !self.will_wake(source) {
            *self = source.clone();
        }
    }
}

#[unstable(feature = "local_waker", issue = "118959")]
impl AsRef<LocalWaker> for Waker {
    fn as_ref(&self) -> &LocalWaker {
        // SAFETY: LocalWaker is just Waker without thread safety
        unsafe { transmute(self) }
    }
}

#[unstable(feature = "local_waker", issue = "118959")]
impl Drop for LocalWaker {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe because `LocalWaker::from_raw` is the only way
        // to initialize `drop` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (self.waker.vtable.drop)(self.waker.data) }
    }
}

#[unstable(feature = "local_waker", issue = "118959")]
impl fmt::Debug for LocalWaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vtable_ptr = self.waker.vtable as *const RawWakerVTable;
        f.debug_struct("LocalWaker")
            .field("data", &self.waker.data)
            .field("vtable", &vtable_ptr)
            .finish()
    }
}

#[unstable(feature = "local_waker", issue = "118959")]
impl !Send for LocalWaker {}
#[unstable(feature = "local_waker", issue = "118959")]
impl !Sync for LocalWaker {}
