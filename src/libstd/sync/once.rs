//! A "once initialization" primitive
//!
//! This primitive is meant to be used to run one-time initialization. An
//! example use case would be for initializing an FFI library.

// A "once" is a relatively simple primitive, and it's also typically provided
// by the OS as well (see `pthread_once` or `InitOnceExecuteOnce`). The OS
// primitives, however, tend to have surprising restrictions, such as the Unix
// one doesn't allow an argument to be passed to the function.
//
// As a result, we end up implementing it ourselves in the standard library.
// This also gives us the opportunity to optimize the implementation a bit which
// should help the fast path on call sites.
//
// This primitive is now just a wrapper around `parking_lot::Once`. Which is faster,
// uses less memory and has fewer constraints than the implementation that was
// here before.

use crate::{fmt, parking_lot};

/// A synchronization primitive which can be used to run a one-time global
/// initialization. Useful for one-time initialization for FFI or related
/// functionality. This type can only be constructed with the [`ONCE_INIT`]
/// value or the equivalent [`Once::new`] constructor.
///
/// [`ONCE_INIT`]: constant.ONCE_INIT.html
/// [`Once::new`]: struct.Once.html#method.new
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
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Once {
    inner: parking_lot::Once,
}

/// State yielded to [`call_once_force`]’s closure parameter. The state can be
/// used to query the poison status of the [`Once`].
///
/// [`call_once_force`]: struct.Once.html#method.call_once_force
/// [`Once`]: struct.Once.html
#[unstable(feature = "once_poison", issue = "33577")]
#[derive(Debug)]
pub struct OnceState {
    poisoned: bool,
}

/// Initialization value for static [`Once`] values.
///
/// [`Once`]: struct.Once.html
///
/// # Examples
///
/// ```
/// use std::sync::{Once, ONCE_INIT};
///
/// static START: Once = ONCE_INIT;
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub const ONCE_INIT: Once = Once::new();

impl Once {
    /// Creates a new `Once` value.
    #[stable(feature = "once_new", since = "1.2.0")]
    pub const fn new() -> Once {
        Once {
            inner: parking_lot::Once::new(),
        }
    }

    /// Performs an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `call_once` has been called,
    /// and otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling thread if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it may not be the closure specified). It is also
    /// guaranteed that any memory writes performed by the executed closure can
    /// be reliably observed by other threads at this point (there is a
    /// happens-before relation between the closure and code executing after the
    /// return).
    ///
    /// If the given closure recursively invokes `call_once` on the same `Once`
    /// instance, it will deadlock.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Once;
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
    /// The closure `f` will only be executed once if this is called
    /// concurrently amongst many threads. If that closure panics, however, then
    /// it will *poison* this `Once` instance, causing all future invocations of
    /// `call_once` to also panic.
    ///
    /// This is similar to [poisoning with mutexes][poison].
    ///
    /// [poison]: struct.Mutex.html#poisoning
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        self.inner.call_once(f);
    }

    /// Performs the same function as [`call_once`] except ignores poisoning.
    ///
    /// Unlike [`call_once`], if this `Once` has been poisoned (i.e., a previous
    /// call to `call_once` or `call_once_force` caused a panic), calling
    /// `call_once_force` will still invoke the closure `f` and will _not_
    /// result in an immediate panic. If `f` panics, the `Once` will remain
    /// in a poison state. If `f` does _not_ panic, the `Once` will no
    /// longer be in a poison state and all future calls to `call_once` or
    /// `call_one_force` will be no-ops.
    ///
    /// The closure `f` is yielded a [`OnceState`] structure which can be used
    /// to query the poison status of the `Once`.
    ///
    /// [`call_once`]: struct.Once.html#method.call_once
    /// [`OnceState`]: struct.OnceState.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_poison)]
    ///
    /// use std::sync::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// // poison the once
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// // poisoning propagates
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| {});
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// // call_once_force will still run and reset the poisoned state
    /// INIT.call_once_force(|state| {
    ///     assert!(state.poisoned());
    /// });
    ///
    /// // once any success happens, we stop propagating the poison
    /// INIT.call_once(|| {});
    /// ```
    #[unstable(feature = "once_poison", issue = "33577")]
    pub fn call_once_force<F>(&self, f: F)
    where
        F: FnOnce(&OnceState),
    {
        self.inner
            .call_once_force(move |state: parking_lot::OnceState| {
                f(&OnceState {
                    poisoned: state.poisoned(),
                });
            });
    }

    /// Returns `true` if some `call_once` call has completed
    /// successfully. Specifically, `is_completed` will return false in
    /// the following situations:
    ///   * `call_once` was not called at all,
    ///   * `call_once` was called, but has not yet completed,
    ///   * the `Once` instance is poisoned
    ///
    /// It is also possible that immediately after `is_completed`
    /// returns false, some other thread finishes executing
    /// `call_once`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(once_is_completed)]
    /// use std::sync::Once;
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
    /// #![feature(once_is_completed)]
    /// use std::sync::Once;
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
    #[unstable(feature = "once_is_completed", issue = "54890")]
    #[inline]
    pub fn is_completed(&self) -> bool {
        self.inner.state().done()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Once {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Once { .. }")
    }
}

impl OnceState {
    /// Returns `true` if the associated [`Once`] was poisoned prior to the
    /// invocation of the closure passed to [`call_once_force`].
    ///
    /// [`call_once_force`]: struct.Once.html#method.call_once_force
    /// [`Once`]: struct.Once.html
    ///
    /// # Examples
    ///
    /// A poisoned `Once`:
    ///
    /// ```
    /// #![feature(once_poison)]
    ///
    /// use std::sync::Once;
    /// use std::thread;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// // poison the once
    /// let handle = thread::spawn(|| {
    ///     INIT.call_once(|| panic!());
    /// });
    /// assert!(handle.join().is_err());
    ///
    /// INIT.call_once_force(|state| {
    ///     assert!(state.poisoned());
    /// });
    /// ```
    ///
    /// An unpoisoned `Once`:
    ///
    /// ```
    /// #![feature(once_poison)]
    ///
    /// use std::sync::Once;
    ///
    /// static INIT: Once = Once::new();
    ///
    /// INIT.call_once_force(|state| {
    ///     assert!(!state.poisoned());
    /// });
    #[unstable(feature = "once_poison", issue = "33577")]
    pub fn poisoned(&self) -> bool {
        self.poisoned
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::Once;
    use crate::panic;
    use crate::sync::mpsc::channel;
    use crate::thread;

    #[test]
    fn smoke_once() {
        static O: Once = Once::new();
        let mut a = 0;
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
        O.call_once(|| a += 1);
        assert_eq!(a, 1);
    }

    #[test]
    fn stampede_once() {
        static O: Once = Once::new();
        static mut RUN: bool = false;

        let (tx, rx) = channel();
        for _ in 0..10 {
            let tx = tx.clone();
            thread::spawn(move || {
                for _ in 0..4 {
                    thread::yield_now()
                }
                unsafe {
                    O.call_once(|| {
                        assert!(!RUN);
                        RUN = true;
                    });
                    assert!(RUN);
                }
                tx.send(()).unwrap();
            });
        }

        unsafe {
            O.call_once(|| {
                assert!(!RUN);
                RUN = true;
            });
            assert!(RUN);
        }

        for _ in 0..10 {
            rx.recv().unwrap();
        }
    }

    #[test]
    fn poison_bad() {
        static O: Once = Once::new();

        // poison the once
        let t = panic::catch_unwind(|| {
            O.call_once(|| panic!());
        });
        assert!(t.is_err());

        // poisoning propagates
        let t = panic::catch_unwind(|| {
            O.call_once(|| {});
        });
        assert!(t.is_err());

        // we can subvert poisoning, however
        let mut called = false;
        O.call_once_force(|p| {
            called = true;
            assert!(p.poisoned())
        });
        assert!(called);

        // once any success happens, we stop propagating the poison
        O.call_once(|| {});
    }

    #[test]
    fn wait_for_force_to_finish() {
        static O: Once = Once::new();

        // poison the once
        let t = panic::catch_unwind(|| {
            O.call_once(|| panic!());
        });
        assert!(t.is_err());

        // make sure someone's waiting inside the once via a force
        let (tx1, rx1) = channel();
        let (tx2, rx2) = channel();
        let t1 = thread::spawn(move || {
            O.call_once_force(|p| {
                assert!(p.poisoned());
                tx1.send(()).unwrap();
                rx2.recv().unwrap();
            });
        });

        rx1.recv().unwrap();

        // put another waiter on the once
        let t2 = thread::spawn(|| {
            let mut called = false;
            O.call_once(|| {
                called = true;
            });
            assert!(!called);
        });

        tx2.send(()).unwrap();

        assert!(t1.join().is_ok());
        assert!(t2.join().is_ok());
    }
}
