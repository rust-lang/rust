use crate::fmt;
use crate::sync::WaitTimeoutResult;
use crate::sync::nonpoison::{MutexGuard, mutex};
use crate::sys::sync as sys;
use crate::time::{Duration, Instant};

/// A Condition Variable
///
/// For more information about condition variables, check out the documentation for the poisoning
/// variant of this type at [`poison::Condvar`].
///
/// # Examples
///
/// Note that this `Condvar` does **not** propagate information about threads that panic while
/// holding a lock. If you need this functionality, see [`poison::Mutex`] and [`poison::Condvar`].
///
/// ```
/// #![feature(nonpoison_mutex)]
/// #![feature(nonpoison_condvar)]
///
/// use std::sync::nonpoison::{Mutex, Condvar};
/// use std::sync::Arc;
/// use std::thread;
///
/// let pair = Arc::new((Mutex::new(false), Condvar::new()));
/// let pair2 = Arc::clone(&pair);
///
/// // Inside of our lock, spawn a new thread, and then wait for it to start.
/// thread::spawn(move || {
///     let (lock, cvar) = &*pair2;
///     let mut started = lock.lock();
///     *started = true;
///     // We notify the condvar that the value has changed.
///     cvar.notify_one();
/// });
///
/// // Wait for the thread to start up.
/// let (lock, cvar) = &*pair;
/// let mut started = lock.lock();
/// while !*started {
///     started = cvar.wait(started);
/// }
/// ```
///
/// [`poison::Mutex`]: crate::sync::poison::Mutex
/// [`poison::Condvar`]: crate::sync::poison::Condvar
#[unstable(feature = "nonpoison_condvar", issue = "134645")]
pub struct Condvar {
    inner: sys::Condvar,
}

impl Condvar {
    /// Creates a new condition variable which is ready to be waited on and
    /// notified.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Condvar;
    ///
    /// let condvar = Condvar::new();
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    #[must_use]
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { inner: sys::Condvar::new() }
    }

    /// Blocks the current thread until this condition variable receives a
    /// notification.
    ///
    /// This function will atomically unlock the mutex specified (represented by
    /// `guard`) and block the current thread. This means that any calls
    /// to [`notify_one`] or [`notify_all`] which happen logically after the
    /// mutex is unlocked are candidates to wake this thread up. When this
    /// function call returns, the lock specified will have been re-acquired.
    ///
    /// Note that this function is susceptible to spurious wakeups. Condition
    /// variables normally have a boolean predicate associated with them, and
    /// the predicate must always be checked each time this function returns to
    /// protect against spurious wakeups.
    ///
    /// # Panics
    ///
    /// This function may [`panic!`] if it is used with more than one mutex
    /// over time.
    ///
    /// [`notify_one`]: Self::notify_one
    /// [`notify_all`]: Self::notify_all
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_mutex)]
    /// #![feature(nonpoison_condvar)]
    ///
    /// use std::sync::nonpoison::{Mutex, Condvar};
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = Arc::clone(&pair);
    ///
    /// thread::spawn(move || {
    ///     let (lock, cvar) = &*pair2;
    ///     let mut started = lock.lock();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let (lock, cvar) = &*pair;
    /// let mut started = lock.lock();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// while !*started {
    ///     started = cvar.wait(started);
    /// }
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        unsafe {
            let lock = mutex::guard_lock(&guard);
            self.inner.wait(lock);
        }
        guard
    }

    /// Blocks the current thread until the provided condition becomes false.
    ///
    /// `condition` is checked immediately; if not met (returns `true`), this
    /// will [`wait`] for the next notification then check again. This repeats
    /// until `condition` returns `false`, in which case this function returns.
    ///
    /// This function will atomically unlock the mutex specified (represented by
    /// `guard`) and block the current thread. This means that any calls
    /// to [`notify_one`] or [`notify_all`] which happen logically after the
    /// mutex is unlocked are candidates to wake this thread up. When this
    /// function call returns, the lock specified will have been re-acquired.
    ///
    /// [`wait`]: Self::wait
    /// [`notify_one`]: Self::notify_one
    /// [`notify_all`]: Self::notify_all
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_mutex)]
    /// #![feature(nonpoison_condvar)]
    ///
    /// use std::sync::nonpoison::{Mutex, Condvar};
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(true), Condvar::new()));
    /// let pair2 = Arc::clone(&pair);
    ///
    /// thread::spawn(move || {
    ///     let (lock, cvar) = &*pair2;
    ///     let mut pending = lock.lock();
    ///     *pending = false;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let (lock, cvar) = &*pair;
    /// // As long as the value inside the `Mutex<bool>` is `true`, we wait.
    /// let _guard = cvar.wait_while(lock.lock(), |pending| { *pending });
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    pub fn wait_while<'a, T, F>(
        &self,
        mut guard: MutexGuard<'a, T>,
        mut condition: F,
    ) -> MutexGuard<'a, T>
    where
        F: FnMut(&mut T) -> bool,
    {
        while condition(&mut *guard) {
            guard = self.wait(guard);
        }
        guard
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to [`wait`] except that
    /// the thread will be blocked for roughly no longer than `dur`. This
    /// method should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that might not cause the maximum
    /// amount of time waited to be precisely `dur`.
    ///
    /// Note that the best effort is made to ensure that the time waited is
    /// measured with a monotonic clock, and not affected by the changes made to
    /// the system time. This function is susceptible to spurious wakeups.
    /// Condition variables normally have a boolean predicate associated with
    /// them, and the predicate must always be checked each time this function
    /// returns to protect against spurious wakeups.  Furthermore, since the timeout
    /// is given relative to the moment this function is called, it needs to be adjusted
    /// when this function is called in a loop. The [`wait_timeout_while`] method
    /// lets you wait with a timeout while a predicate is true, taking care of all these concerns.
    ///
    /// The returned [`WaitTimeoutResult`] value indicates if the timeout is
    /// known to have elapsed.
    ///
    /// Like [`wait`], the lock specified will be re-acquired when this function
    /// returns, regardless of whether the timeout elapsed or not.
    ///
    /// [`wait`]: Self::wait
    /// [`wait_timeout_while`]: Self::wait_timeout_while
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_mutex)]
    /// #![feature(nonpoison_condvar)]
    ///
    /// use std::sync::nonpoison::{Mutex, Condvar};
    /// use std::sync::Arc;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = Arc::clone(&pair);
    ///
    /// thread::spawn(move || {
    ///     let (lock, cvar) = &*pair2;
    ///     let mut started = lock.lock();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // wait for the thread to start up
    /// let (lock, cvar) = &*pair;
    /// let mut started = lock.lock();
    /// // as long as the value inside the `Mutex<bool>` is `false`, we wait
    /// loop {
    ///     let result = cvar.wait_timeout(started, Duration::from_millis(10));
    ///     // 10 milliseconds have passed, or maybe the value changed!
    ///     started = result.0;
    ///     if *started == true {
    ///         // We received the notification and the value has been updated, we can leave.
    ///         break
    ///     }
    /// }
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    pub fn wait_timeout<'a, T>(
        &self,
        guard: MutexGuard<'a, T>,
        dur: Duration,
    ) -> (MutexGuard<'a, T>, WaitTimeoutResult) {
        let success = unsafe {
            let lock = mutex::guard_lock(&guard);
            self.inner.wait_timeout(lock, dur)
        };
        (guard, WaitTimeoutResult(!success))
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to [`wait_while`] except
    /// that the thread will be blocked for roughly no longer than `dur`. This
    /// method should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that might not cause the maximum
    /// amount of time waited to be precisely `dur`.
    ///
    /// Note that the best effort is made to ensure that the time waited is
    /// measured with a monotonic clock, and not affected by the changes made to
    /// the system time.
    ///
    /// The returned [`WaitTimeoutResult`] value indicates if the timeout is
    /// known to have elapsed without the condition being met.
    ///
    /// Like [`wait_while`], the lock specified will be re-acquired when this
    /// function returns, regardless of whether the timeout elapsed or not.
    ///
    /// [`wait_while`]: Self::wait_while
    /// [`wait_timeout`]: Self::wait_timeout
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_mutex)]
    /// #![feature(nonpoison_condvar)]
    ///
    /// use std::sync::nonpoison::{Mutex, Condvar};
    /// use std::sync::Arc;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let pair = Arc::new((Mutex::new(true), Condvar::new()));
    /// let pair2 = Arc::clone(&pair);
    ///
    /// thread::spawn(move || {
    ///     let (lock, cvar) = &*pair2;
    ///     let mut pending = lock.lock();
    ///     *pending = false;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // wait for the thread to start up
    /// let (lock, cvar) = &*pair;
    /// let result = cvar.wait_timeout_while(
    ///     lock.lock(),
    ///     Duration::from_millis(100),
    ///     |&mut pending| pending,
    /// );
    /// if result.1.timed_out() {
    ///     // timed-out without the condition ever evaluating to false.
    /// }
    /// // access the locked mutex via result.0
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    pub fn wait_timeout_while<'a, T, F>(
        &self,
        mut guard: MutexGuard<'a, T>,
        dur: Duration,
        mut condition: F,
    ) -> (MutexGuard<'a, T>, WaitTimeoutResult)
    where
        F: FnMut(&mut T) -> bool,
    {
        let start = Instant::now();
        loop {
            if !condition(&mut *guard) {
                return (guard, WaitTimeoutResult(false));
            }
            let timeout = match dur.checked_sub(start.elapsed()) {
                Some(timeout) => timeout,
                None => return (guard, WaitTimeoutResult(true)),
            };
            guard = self.wait_timeout(guard, timeout).0;
        }
    }

    /// Wakes up one blocked thread on this condvar.
    ///
    /// If there is a blocked thread on this condition variable, then it will
    /// be woken up from its call to [`wait`] or [`wait_timeout`]. Calls to
    /// `notify_one` are not buffered in any way.
    ///
    /// To wake up all threads, see [`notify_all`].
    ///
    /// [`wait`]: Self::wait
    /// [`wait_timeout`]: Self::wait_timeout
    /// [`notify_all`]: Self::notify_all
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_mutex)]
    /// #![feature(nonpoison_condvar)]
    ///
    /// use std::sync::nonpoison::{Mutex, Condvar};
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = Arc::clone(&pair);
    ///
    /// thread::spawn(move || {
    ///     let (lock, cvar) = &*pair2;
    ///     let mut started = lock.lock();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let (lock, cvar) = &*pair;
    /// let mut started = lock.lock();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// while !*started {
    ///     started = cvar.wait(started);
    /// }
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    pub fn notify_one(&self) {
        self.inner.notify_one()
    }

    /// Wakes up all blocked threads on this condvar.
    ///
    /// This method will ensure that any current waiters on the condition
    /// variable are awoken. Calls to `notify_all()` are not buffered in any
    /// way.
    ///
    /// To wake up only one thread, see [`notify_one`].
    ///
    /// [`notify_one`]: Self::notify_one
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(nonpoison_mutex)]
    /// #![feature(nonpoison_condvar)]
    ///
    /// use std::sync::nonpoison::{Mutex, Condvar};
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = Arc::clone(&pair);
    ///
    /// thread::spawn(move || {
    ///     let (lock, cvar) = &*pair2;
    ///     let mut started = lock.lock();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_all();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let (lock, cvar) = &*pair;
    /// let mut started = lock.lock();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// while !*started {
    ///     started = cvar.wait(started);
    /// }
    /// ```
    #[unstable(feature = "nonpoison_condvar", issue = "134645")]
    pub fn notify_all(&self) {
        self.inner.notify_all()
    }
}

#[unstable(feature = "nonpoison_condvar", issue = "134645")]
impl fmt::Debug for Condvar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Condvar").finish_non_exhaustive()
    }
}

#[unstable(feature = "nonpoison_condvar", issue = "134645")]
impl Default for Condvar {
    /// Creates a `Condvar` which is ready to be waited on and notified.
    fn default() -> Condvar {
        Condvar::new()
    }
}
