#![stable(feature = "futures_api", since = "1.36.0")]

use crate::fmt;
use crate::marker::{PhantomData, Unpin};

/// A `RawWaker` allows the implementor of a task executor to create a [`Waker`]
/// which provides customized wakeup behavior.
///
/// [vtable]: https://en.wikipedia.org/wiki/Virtual_method_table
///
/// It consists of a data pointer and a [virtual function pointer table (vtable)][vtable]
/// that customizes the behavior of the `RawWaker`.
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

    /// This function gets called when a [`RawWaker`] gets dropped.
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
    /// # `clone`
    ///
    /// This function will be called when the [`RawWaker`] gets cloned, e.g. when
    /// the [`Waker`] in which the [`RawWaker`] is stored gets cloned.
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
    /// This function gets called when a [`RawWaker`] gets dropped.
    ///
    /// The implementation of this function must make sure to release any
    /// resources that are associated with this instance of a [`RawWaker`] and
    /// associated task.
    #[rustc_promotable]
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_const_stable(feature = "futures_api", since = "1.36.0")]
    #[rustc_allow_const_fn_unstable(const_fn_fn_ptr_basics)]
    pub const fn new(
        clone: unsafe fn(*const ()) -> RawWaker,
        wake: unsafe fn(*const ()),
        wake_by_ref: unsafe fn(*const ()),
        drop: unsafe fn(*const ()),
    ) -> Self {
        Self { clone, wake, wake_by_ref, drop }
    }
}

/// The `Context` of an asynchronous task.
///
/// Currently, `Context` only serves to provide access to a `&Waker`
/// which can be used to wake the current task.
#[stable(feature = "futures_api", since = "1.36.0")]
pub struct Context<'a> {
    waker: &'a Waker,
    // Ensure we future-proof against variance changes by forcing
    // the lifetime to be invariant (argument-position lifetimes
    // are contravariant while return-position lifetimes are
    // covariant).
    _marker: PhantomData<fn(&'a ()) -> &'a ()>,
}

impl<'a> Context<'a> {
    /// Create a new `Context` from a `&Waker`.
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[must_use]
    #[inline]
    pub fn from_waker(waker: &'a Waker) -> Self {
        Context { waker, _marker: PhantomData }
    }

    /// Returns a reference to the `Waker` for the current task.
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[inline]
    pub fn waker(&self) -> &'a Waker {
        &self.waker
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl fmt::Debug for Context<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").field("waker", &self.waker).finish()
    }
}

/// A `Waker` is a handle for waking up a task by notifying its executor that it
/// is ready to be run.
///
/// This handle encapsulates a [`RawWaker`] instance, which defines the
/// executor-specific wakeup behavior.
///
/// Implements [`Clone`], [`Send`], and [`Sync`].
#[repr(transparent)]
#[stable(feature = "futures_api", since = "1.36.0")]
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
    /// Wake up the task associated with this `Waker`.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn wake(self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.
        let wake = self.waker.vtable.wake;
        let data = self.waker.data;

        // Don't call `drop` -- the waker will be consumed by `wake`.
        crate::mem::forget(self);

        // SAFETY: This is safe because `Waker::from_raw` is the only way
        // to initialize `wake` and `data` requiring the user to acknowledge
        // that the contract of `RawWaker` is upheld.
        unsafe { (wake)(data) };
    }

    /// Wake up the task associated with this `Waker` without consuming the `Waker`.
    ///
    /// This is similar to `wake`, but may be slightly less efficient in the case
    /// where an owned `Waker` is available. This method should be preferred to
    /// calling `waker.clone().wake()`.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn wake_by_ref(&self) {
        // The actual wakeup call is delegated through a virtual function call
        // to the implementation which is defined by the executor.

        // SAFETY: see `wake`
        unsafe { (self.waker.vtable.wake_by_ref)(self.waker.data) }
    }

    /// Returns `true` if this `Waker` and another `Waker` have awoken the same task.
    ///
    /// This function works on a best-effort basis, and may return false even
    /// when the `Waker`s would awaken the same task. However, if this function
    /// returns `true`, it is guaranteed that the `Waker`s will awaken the same task.
    ///
    /// This function is primarily used for optimization purposes.
    #[inline]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub fn will_wake(&self, other: &Waker) -> bool {
        self.waker == other.waker
    }

    /// Creates a new `Waker` from [`RawWaker`].
    ///
    /// The behavior of the returned `Waker` is undefined if the contract defined
    /// in [`RawWaker`]'s and [`RawWakerVTable`]'s documentation is not upheld.
    /// Therefore this method is unsafe.
    #[inline]
    #[must_use]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub unsafe fn from_raw(waker: RawWaker) -> Waker {
        Waker { waker }
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
