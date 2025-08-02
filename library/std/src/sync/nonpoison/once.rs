use crate::fmt;
use crate::sys::sync as sys;

/// A low-level synchronization primitive for one-time global execution that ignores poisoning.
///
/// For more information about `Once`, check out the documentation for the poisoning variant (which
/// can be found at [`poison::Once`]).
///
/// [`poison::Once`]: crate::sync::poison::Once
///
/// # Examples
///
/// ```
/// use std::sync::Once;
///
/// static START: Once = Once::new();
///
/// START.call_once(|| {
///     // run initialization here
/// });
/// ```
#[unstable(feature = "nonpoison_once", issue = "134645")]
pub struct Once {
    inner: sys::Once,
}

impl Once {
    /// Creates a new `Once` value.
    #[inline]
    #[unstable(feature = "nonpoison_once", issue = "134645")]
    #[must_use]
    pub const fn new() -> Once {
        Once { inner: sys::Once::new() }
    }

    /// Performs an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `call_once` has been called,
    /// and otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it might not be the closure specified). It is also
    /// guaranteed that any memory writes performed by the executed closure can
    /// be reliably observed by other threads at this point (there is a
    /// happens-before relation between the closure and code executing after the
    /// return).
    ///
    /// If the given closure recursively invokes `call_once` on the same [`Once`]
    /// instance, the exact behavior is not specified: allowed outcomes are
    /// a panic or a deadlock.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_once)]
    ///
    /// use std::sync::nonpoison::Once;
    ///
    /// static mut VAL: usize = 0;
    /// static INIT: Once = Once::new();
    ///
    /// // Accessing a `static mut` is unsafe much of the time, but if we do so
    /// // in a synchronized fashion (e.g., write once or read all) then we're
    /// // good to go!
    /// //
    /// // This function will only call `expensive_computation` once, and will
    /// // otherwise always return the value returned from the first invocation.
    /// fn get_cached_val() -> usize {
    ///     unsafe {
    ///         INIT.call_once(|| {
    ///             VAL = expensive_computation();
    ///         });
    ///         VAL
    ///     }
    /// }
    ///
    /// fn expensive_computation() -> usize {
    ///     // ...
    /// # 2
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The closure `f` will only be executed once even if this is called concurrently amongst many
    /// threads. If the closure panics, the calling thread will panic and the `Once` will remain in
    /// an incompleted state.
    ///
    /// In contrast to the [`poison::Once`] variant, all calls to `call_once` will ignore panics in
    /// other threads. This method is identical to the [`poison::Once::call_once_force`] method.
    ///
    /// If you need observability into whether any threads have panicked while calling `call_once`,
    /// see [`poison::Once`].
    ///
    /// [`poison::Once`]: crate::sync::poison::Once
    /// [`poison::Once::call_once_force`]: crate::sync::poison::Once::call_once_force
    ///
    /// ```
    /// #![feature(nonpoison_once)]
    ///
    /// use std::sync::nonpoison::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// // Panic during `call_once`.
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// // `call_once` will still run from a different thread.
    /// INIT.call_once(|| {
    ///     assert_eq!(2 + 2, 4);
    /// });
    /// ```
    #[inline]
    #[unstable(feature = "nonpoison_once", issue = "134645")]
    #[track_caller]
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        // Fast path check.
        if self.inner.is_completed() {
            return;
        }

        let mut f = Some(f);
        self.inner.call(true, &mut |_| f.take().unwrap()());
    }

    /// Returns `true` if some [`call_once()`] call has completed
    /// successfully. Specifically, `is_completed` will return false in
    /// the following situations:
    ///   * [`call_once()`] was not called at all,
    ///   * [`call_once()`] was called, but has not yet completed
    ///
    /// This function returning `false` does not mean that [`Once`] has not been
    /// executed. For example, it may have been executed in the time between
    /// when `is_completed` starts executing and when it returns, in which case
    /// the `false` return value would be stale (but still permissible).
    ///
    /// [`call_once()`]: Once::call_once
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_once)]
    ///
    /// use std::sync::nonpoison::Once;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// assert_eq!(INIT.is_completed(), false);
    /// INIT.call_once(|| {
    ///     assert_eq!(INIT.is_completed(), false);
    /// });
    /// assert_eq!(INIT.is_completed(), true);
    /// ```
    ///
    /// ```
    /// #![feature(nonpoison_once)]
    ///
    /// use std::sync::nonpoison::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// assert_eq!(INIT.is_completed(), false);
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    /// assert_eq!(INIT.is_completed(), false);
    /// ```
    #[inline]
    #[unstable(feature = "nonpoison_once", issue = "134645")]
    pub fn is_completed(&self) -> bool {
        self.inner.is_completed()
    }

    /// Blocks the current thread until initialization has completed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Once;
    /// use std::thread;
    ///
    /// static READY: Once = Once::new();
    ///
    /// let thread = thread::spawn(|| {
    ///     READY.wait();
    ///     println!("everything is ready");
    /// });
    ///
    /// READY.call_once(|| println!("performing setup"));
    /// ```
    ///
    /// This function will continue to block even if a thread initializing via [`call_once()`] has
    /// panicked. This behavior is identical to the [`poison::Once::wait_force`] method.
    ///
    /// [`call_once()`]: Once::call_once
    /// [`poison::Once::wait_force`]: crate::sync::poison::Once::wait_force
    #[unstable(feature = "nonpoison_once", issue = "134645")]
    pub fn wait(&self) {
        if !self.inner.is_completed() {
            self.inner.wait(true);
        }
    }
}

#[unstable(feature = "nonpoison_once", issue = "134645")]
impl fmt::Debug for Once {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Once").finish_non_exhaustive()
    }
}
