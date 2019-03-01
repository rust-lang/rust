use crate::fmt;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::{mutex, MutexGuard, PoisonError};
use crate::sys_common::condvar as sys;
use crate::sys_common::mutex as sys_mutex;
use crate::sys_common::poison::{self, LockResult};
use crate::time::{Duration, Instant};

/// A type indicating whether a timed wait on a condition variable returned
/// due to a time out or not.
///
/// It is returned by the [`wait_timeout`] method.
///
/// [`wait_timeout`]: struct.Condvar.html#method.wait_timeout
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[stable(feature = "wait_timeout", since = "1.5.0")]
pub struct WaitTimeoutResult(bool);

impl WaitTimeoutResult {
    /// Returns `true` if the wait was known to have timed out.
    ///
    /// # Examples
    ///
    /// This example spawns a thread which will update the boolean value and
    /// then wait 100 milliseconds before notifying the condvar.
    ///
    /// The main thread will wait with a timeout on the condvar and then leave
    /// once the boolean has been updated and notified.
    ///
    /// ```
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///
    ///     // Let's wait 20 milliseconds before notifying the condvar.
    ///     thread::sleep(Duration::from_millis(20));
    ///
    ///     let mut started = lock.lock().unwrap();
    ///     // We update the boolean value.
    ///     *started = true;
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let &(ref lock, ref cvar) = &*pair;
    /// let mut started = lock.lock().unwrap();
    /// loop {
    ///     // Let's put a timeout on the condvar's wait.
    ///     let result = cvar.wait_timeout(started, Duration::from_millis(10)).unwrap();
    ///     // 10 milliseconds have passed, or maybe the value changed!
    ///     started = result.0;
    ///     if *started == true {
    ///         // We received the notification and the value has been updated, we can leave.
    ///         break
    ///     }
    /// }
    /// ```
    #[stable(feature = "wait_timeout", since = "1.5.0")]
    pub fn timed_out(&self) -> bool {
        self.0
    }
}

/// A Condition Variable
///
/// Condition variables represent the ability to block a thread such that it
/// consumes no CPU time while waiting for an event to occur. Condition
/// variables are typically associated with a boolean predicate (a condition)
/// and a mutex. The predicate is always verified inside of the mutex before
/// determining that a thread must block.
///
/// Functions in this module will block the current **thread** of execution and
/// are bindings to system-provided condition variables where possible. Note
/// that this module places one additional restriction over the system condition
/// variables: each condvar can be used with precisely one mutex at runtime. Any
/// attempt to use multiple mutexes on the same condition variable will result
/// in a runtime panic. If this is not desired, then the unsafe primitives in
/// `sys` do not have this restriction but may result in undefined behavior.
///
/// # Examples
///
/// ```
/// use std::sync::{Arc, Mutex, Condvar};
/// use std::thread;
///
/// let pair = Arc::new((Mutex::new(false), Condvar::new()));
/// let pair2 = pair.clone();
///
/// // Inside of our lock, spawn a new thread, and then wait for it to start.
/// thread::spawn(move|| {
///     let &(ref lock, ref cvar) = &*pair2;
///     let mut started = lock.lock().unwrap();
///     *started = true;
///     // We notify the condvar that the value has changed.
///     cvar.notify_one();
/// });
///
/// // Wait for the thread to start up.
/// let &(ref lock, ref cvar) = &*pair;
/// let mut started = lock.lock().unwrap();
/// while !*started {
///     started = cvar.wait(started).unwrap();
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Condvar {
    inner: Box<sys::Condvar>,
    mutex: AtomicUsize,
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Condvar {
        let mut c = Condvar {
            inner: box sys::Condvar::new(),
            mutex: AtomicUsize::new(0),
        };
        unsafe {
            c.inner.init();
        }
        c
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
    /// # Errors
    ///
    /// This function will return an error if the mutex being waited on is
    /// poisoned when this thread re-acquires the lock. For more information,
    /// see information about [poisoning] on the [`Mutex`] type.
    ///
    /// # Panics
    ///
    /// This function will [`panic!`] if it is used with more than one mutex
    /// over time. Each condition variable is dynamically bound to exactly one
    /// mutex to ensure defined behavior across platforms. If this functionality
    /// is not desired, then unsafe primitives in `sys` are provided.
    ///
    /// [`notify_one`]: #method.notify_one
    /// [`notify_all`]: #method.notify_all
    /// [poisoning]: ../sync/struct.Mutex.html#poisoning
    /// [`Mutex`]: ../sync/struct.Mutex.html
    /// [`panic!`]: ../../std/macro.panic.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let &(ref lock, ref cvar) = &*pair;
    /// let mut started = lock.lock().unwrap();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// while !*started {
    ///     started = cvar.wait(started).unwrap();
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>)
                       -> LockResult<MutexGuard<'a, T>> {
        let poisoned = unsafe {
            let lock = mutex::guard_lock(&guard);
            self.verify(lock);
            self.inner.wait(lock);
            mutex::guard_poison(&guard).get()
        };
        if poisoned {
            Err(PoisonError::new(guard))
        } else {
            Ok(guard)
        }
    }

    /// Blocks the current thread until this condition variable receives a
    /// notification and the required condition is met. Spurious wakeups are
    /// ignored and this function will only return once the condition has been
    /// met.
    ///
    /// This function will atomically unlock the mutex specified (represented by
    /// `guard`) and block the current thread. This means that any calls
    /// to [`notify_one`] or [`notify_all`] which happen logically after the
    /// mutex is unlocked are candidates to wake this thread up. When this
    /// function call returns, the lock specified will have been re-acquired.
    ///
    /// # Errors
    ///
    /// This function will return an error if the mutex being waited on is
    /// poisoned when this thread re-acquires the lock. For more information,
    /// see information about [poisoning] on the [`Mutex`] type.
    ///
    /// [`notify_one`]: #method.notify_one
    /// [`notify_all`]: #method.notify_all
    /// [poisoning]: ../sync/struct.Mutex.html#poisoning
    /// [`Mutex`]: ../sync/struct.Mutex.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(wait_until)]
    ///
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let &(ref lock, ref cvar) = &*pair;
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// let _guard = cvar.wait_until(lock.lock().unwrap(), |started| { *started }).unwrap();
    /// ```
    #[unstable(feature = "wait_until", issue = "47960")]
    pub fn wait_until<'a, T, F>(&self, mut guard: MutexGuard<'a, T>,
                                mut condition: F)
                                -> LockResult<MutexGuard<'a, T>>
                                where F: FnMut(&mut T) -> bool {
        while !condition(&mut *guard) {
            guard = self.wait(guard)?;
        }
        Ok(guard)
    }


    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to [`wait`]
    /// except that the thread will be blocked for roughly no longer
    /// than `ms` milliseconds. This method should not be used for
    /// precise timing due to anomalies such as preemption or platform
    /// differences that may not cause the maximum amount of time
    /// waited to be precisely `ms`.
    ///
    /// Note that the best effort is made to ensure that the time waited is
    /// measured with a monotonic clock, and not affected by the changes made to
    /// the system time.
    ///
    /// The returned boolean is `false` only if the timeout is known
    /// to have elapsed.
    ///
    /// Like [`wait`], the lock specified will be re-acquired when this function
    /// returns, regardless of whether the timeout elapsed or not.
    ///
    /// [`wait`]: #method.wait
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let &(ref lock, ref cvar) = &*pair;
    /// let mut started = lock.lock().unwrap();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// loop {
    ///     let result = cvar.wait_timeout_ms(started, 10).unwrap();
    ///     // 10 milliseconds have passed, or maybe the value changed!
    ///     started = result.0;
    ///     if *started == true {
    ///         // We received the notification and the value has been updated, we can leave.
    ///         break
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.6.0", reason = "replaced by `std::sync::Condvar::wait_timeout`")]
    pub fn wait_timeout_ms<'a, T>(&self, guard: MutexGuard<'a, T>, ms: u32)
                                  -> LockResult<(MutexGuard<'a, T>, bool)> {
        let res = self.wait_timeout(guard, Duration::from_millis(ms as u64));
        poison::map_result(res, |(a, b)| {
            (a, !b.timed_out())
        })
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to [`wait`] except that
    /// the thread will be blocked for roughly no longer than `dur`. This
    /// method should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that may not cause the maximum
    /// amount of time waited to be precisely `dur`.
    ///
    /// Note that the best effort is made to ensure that the time waited is
    /// measured with a monotonic clock, and not affected by the changes made to
    /// the system time. This function is susceptible to spurious wakeups.
    /// Condition variables normally have a boolean predicate associated with
    /// them, and the predicate must always be checked each time this function
    /// returns to protect against spurious wakeups. Additionally, it is
    /// typically desirable for the time-out to not exceed some duration in
    /// spite of spurious wakes, thus the sleep-duration is decremented by the
    /// amount slept. Alternatively, use the `wait_timeout_until` method
    /// to wait until a condition is met with a total time-out regardless
    /// of spurious wakes.
    ///
    /// The returned [`WaitTimeoutResult`] value indicates if the timeout is
    /// known to have elapsed.
    ///
    /// Like [`wait`], the lock specified will be re-acquired when this function
    /// returns, regardless of whether the timeout elapsed or not.
    ///
    /// [`wait`]: #method.wait
    /// [`wait_timeout_until`]: #method.wait_timeout_until
    /// [`WaitTimeoutResult`]: struct.WaitTimeoutResult.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // wait for the thread to start up
    /// let &(ref lock, ref cvar) = &*pair;
    /// let mut started = lock.lock().unwrap();
    /// // as long as the value inside the `Mutex<bool>` is `false`, we wait
    /// loop {
    ///     let result = cvar.wait_timeout(started, Duration::from_millis(10)).unwrap();
    ///     // 10 milliseconds have passed, or maybe the value changed!
    ///     started = result.0;
    ///     if *started == true {
    ///         // We received the notification and the value has been updated, we can leave.
    ///         break
    ///     }
    /// }
    /// ```
    #[stable(feature = "wait_timeout", since = "1.5.0")]
    pub fn wait_timeout<'a, T>(&self, guard: MutexGuard<'a, T>,
                               dur: Duration)
                               -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)> {
        let (poisoned, result) = unsafe {
            let lock = mutex::guard_lock(&guard);
            self.verify(lock);
            let success = self.inner.wait_timeout(lock, dur);
            (mutex::guard_poison(&guard).get(), WaitTimeoutResult(!success))
        };
        if poisoned {
            Err(PoisonError::new((guard, result)))
        } else {
            Ok((guard, result))
        }
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration. Spurious wakes will not cause this function to
    /// return.
    ///
    /// The semantics of this function are equivalent to [`wait_until`] except
    /// that the thread will be blocked for roughly no longer than `dur`. This
    /// method should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that may not cause the maximum
    /// amount of time waited to be precisely `dur`.
    ///
    /// Note that the best effort is made to ensure that the time waited is
    /// measured with a monotonic clock, and not affected by the changes made to
    /// the system time.
    ///
    /// The returned [`WaitTimeoutResult`] value indicates if the timeout is
    /// known to have elapsed without the condition being met.
    ///
    /// Like [`wait_until`], the lock specified will be re-acquired when this
    /// function returns, regardless of whether the timeout elapsed or not.
    ///
    /// [`wait_until`]: #method.wait_until
    /// [`wait_timeout`]: #method.wait_timeout
    /// [`WaitTimeoutResult`]: struct.WaitTimeoutResult.html
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(wait_timeout_until)]
    ///
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // wait for the thread to start up
    /// let &(ref lock, ref cvar) = &*pair;
    /// let result = cvar.wait_timeout_until(
    ///     lock.lock().unwrap(),
    ///     Duration::from_millis(100),
    ///     |&mut started| started,
    /// ).unwrap();
    /// if result.1.timed_out() {
    ///     // timed-out without the condition ever evaluating to true.
    /// }
    /// // access the locked mutex via result.0
    /// ```
    #[unstable(feature = "wait_timeout_until", issue = "47960")]
    pub fn wait_timeout_until<'a, T, F>(&self, mut guard: MutexGuard<'a, T>,
                                        dur: Duration, mut condition: F)
                                        -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)>
                                        where F: FnMut(&mut T) -> bool {
        let start = Instant::now();
        loop {
            if condition(&mut *guard) {
                return Ok((guard, WaitTimeoutResult(false)));
            }
            let timeout = match dur.checked_sub(start.elapsed()) {
                Some(timeout) => timeout,
                None => return Ok((guard, WaitTimeoutResult(true))),
            };
            guard = self.wait_timeout(guard, timeout)?.0;
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
    /// [`wait`]: #method.wait
    /// [`wait_timeout`]: #method.wait_timeout
    /// [`notify_all`]: #method.notify_all
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_one();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let &(ref lock, ref cvar) = &*pair;
    /// let mut started = lock.lock().unwrap();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// while !*started {
    ///     started = cvar.wait(started).unwrap();
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn notify_one(&self) {
        unsafe { self.inner.notify_one() }
    }

    /// Wakes up all blocked threads on this condvar.
    ///
    /// This method will ensure that any current waiters on the condition
    /// variable are awoken. Calls to `notify_all()` are not buffered in any
    /// way.
    ///
    /// To wake up only one thread, see [`notify_one`].
    ///
    /// [`notify_one`]: #method.notify_one
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex, Condvar};
    /// use std::thread;
    ///
    /// let pair = Arc::new((Mutex::new(false), Condvar::new()));
    /// let pair2 = pair.clone();
    ///
    /// thread::spawn(move|| {
    ///     let &(ref lock, ref cvar) = &*pair2;
    ///     let mut started = lock.lock().unwrap();
    ///     *started = true;
    ///     // We notify the condvar that the value has changed.
    ///     cvar.notify_all();
    /// });
    ///
    /// // Wait for the thread to start up.
    /// let &(ref lock, ref cvar) = &*pair;
    /// let mut started = lock.lock().unwrap();
    /// // As long as the value inside the `Mutex<bool>` is `false`, we wait.
    /// while !*started {
    ///     started = cvar.wait(started).unwrap();
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn notify_all(&self) {
        unsafe { self.inner.notify_all() }
    }

    fn verify(&self, mutex: &sys_mutex::Mutex) {
        let addr = mutex as *const _ as usize;
        match self.mutex.compare_and_swap(0, addr, Ordering::SeqCst) {
            // If we got out 0, then we have successfully bound the mutex to
            // this cvar.
            0 => {}

            // If we get out a value that's the same as `addr`, then someone
            // already beat us to the punch.
            n if n == addr => {}

            // Anything else and we're using more than one mutex on this cvar,
            // which is currently disallowed.
            _ => panic!("attempted to use a condition variable with two \
                         mutexes"),
        }
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Condvar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Condvar { .. }")
    }
}

#[stable(feature = "condvar_default", since = "1.10.0")]
impl Default for Condvar {
    /// Creates a `Condvar` which is ready to be waited on and notified.
    fn default() -> Condvar {
        Condvar::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Drop for Condvar {
    fn drop(&mut self) {
        unsafe { self.inner.destroy() }
    }
}

#[cfg(test)]
mod tests {
    /// #![feature(wait_until)]
    use crate::sync::mpsc::channel;
    use crate::sync::{Condvar, Mutex, Arc};
    use crate::sync::atomic::{AtomicBool, Ordering};
    use crate::thread;
    use crate::time::Duration;
    use crate::u64;

    #[test]
    fn smoke() {
        let c = Condvar::new();
        c.notify_one();
        c.notify_all();
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn notify_one() {
        let m = Arc::new(Mutex::new(()));
        let m2 = m.clone();
        let c = Arc::new(Condvar::new());
        let c2 = c.clone();

        let g = m.lock().unwrap();
        let _t = thread::spawn(move|| {
            let _g = m2.lock().unwrap();
            c2.notify_one();
        });
        let g = c.wait(g).unwrap();
        drop(g);
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn notify_all() {
        const N: usize = 10;

        let data = Arc::new((Mutex::new(0), Condvar::new()));
        let (tx, rx) = channel();
        for _ in 0..N {
            let data = data.clone();
            let tx = tx.clone();
            thread::spawn(move|| {
                let &(ref lock, ref cond) = &*data;
                let mut cnt = lock.lock().unwrap();
                *cnt += 1;
                if *cnt == N {
                    tx.send(()).unwrap();
                }
                while *cnt != 0 {
                    cnt = cond.wait(cnt).unwrap();
                }
                tx.send(()).unwrap();
            });
        }
        drop(tx);

        let &(ref lock, ref cond) = &*data;
        rx.recv().unwrap();
        let mut cnt = lock.lock().unwrap();
        *cnt = 0;
        cond.notify_all();
        drop(cnt);

        for _ in 0..N {
            rx.recv().unwrap();
        }
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn wait_until() {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair2 = pair.clone();

        // Inside of our lock, spawn a new thread, and then wait for it to start.
        thread::spawn(move|| {
            let &(ref lock, ref cvar) = &*pair2;
            let mut started = lock.lock().unwrap();
            *started = true;
            // We notify the condvar that the value has changed.
            cvar.notify_one();
        });

        // Wait for the thread to start up.
        let &(ref lock, ref cvar) = &*pair;
        let guard = cvar.wait_until(lock.lock().unwrap(), |started| {
            *started
        });
        assert!(*guard.unwrap());
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn wait_timeout_wait() {
        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        loop {
            let g = m.lock().unwrap();
            let (_g, no_timeout) = c.wait_timeout(g, Duration::from_millis(1)).unwrap();
            // spurious wakeups mean this isn't necessarily true
            // so execute test again, if not timeout
            if !no_timeout.timed_out() {
                continue;
            }

            break;
        }
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn wait_timeout_until_wait() {
        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        let g = m.lock().unwrap();
        let (_g, wait) = c.wait_timeout_until(g, Duration::from_millis(1), |_| { false }).unwrap();
        // no spurious wakeups. ensure it timed-out
        assert!(wait.timed_out());
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn wait_timeout_until_instant_satisfy() {
        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        let g = m.lock().unwrap();
        let (_g, wait) = c.wait_timeout_until(g, Duration::from_millis(0), |_| { true }).unwrap();
        // ensure it didn't time-out even if we were not given any time.
        assert!(!wait.timed_out());
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn wait_timeout_until_wake() {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair_copy = pair.clone();

        let &(ref m, ref c) = &*pair;
        let g = m.lock().unwrap();
        let _t = thread::spawn(move || {
            let &(ref lock, ref cvar) = &*pair_copy;
            let mut started = lock.lock().unwrap();
            thread::sleep(Duration::from_millis(1));
            *started = true;
            cvar.notify_one();
        });
        let (g2, wait) = c.wait_timeout_until(g, Duration::from_millis(u64::MAX), |&mut notified| {
            notified
        }).unwrap();
        // ensure it didn't time-out even if we were not given any time.
        assert!(!wait.timed_out());
        assert!(*g2);
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn wait_timeout_wake() {
        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        loop {
            let g = m.lock().unwrap();

            let c2 = c.clone();
            let m2 = m.clone();

            let notified = Arc::new(AtomicBool::new(false));
            let notified_copy = notified.clone();

            let t = thread::spawn(move || {
                let _g = m2.lock().unwrap();
                thread::sleep(Duration::from_millis(1));
                notified_copy.store(true, Ordering::SeqCst);
                c2.notify_one();
            });
            let (g, timeout_res) = c.wait_timeout(g, Duration::from_millis(u64::MAX)).unwrap();
            assert!(!timeout_res.timed_out());
            // spurious wakeups mean this isn't necessarily true
            // so execute test again, if not notified
            if !notified.load(Ordering::SeqCst) {
                t.join().unwrap();
                continue;
            }
            drop(g);

            t.join().unwrap();

            break;
        }
    }

    #[test]
    #[should_panic]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn two_mutexes() {
        let m = Arc::new(Mutex::new(()));
        let m2 = m.clone();
        let c = Arc::new(Condvar::new());
        let c2 = c.clone();

        let mut g = m.lock().unwrap();
        let _t = thread::spawn(move|| {
            let _g = m2.lock().unwrap();
            c2.notify_one();
        });
        g = c.wait(g).unwrap();
        drop(g);

        let m = Mutex::new(());
        let _ = c.wait(m.lock().unwrap()).unwrap();
    }
}
