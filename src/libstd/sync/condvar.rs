// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use sync::atomic::{AtomicUsize, Ordering};
use sync::{mutex, MutexGuard, PoisonError};
use sys_common::condvar as sys;
use sys_common::mutex as sys_mutex;
use sys_common::poison::{self, LockResult};
use time::{Instant, Duration};

/// A type indicating whether a timed wait on a condition variable returned
/// due to a time out or not.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
#[stable(feature = "wait_timeout", since = "1.5.0")]
pub struct WaitTimeoutResult(bool);

impl WaitTimeoutResult {
    /// Returns whether the wait was known to have timed out.
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
/// determining that thread must block.
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
/// // Inside of our lock, spawn a new thread, and then wait for it to start
/// thread::spawn(move|| {
///     let &(ref lock, ref cvar) = &*pair2;
///     let mut started = lock.lock().unwrap();
///     *started = true;
///     cvar.notify_one();
/// });
///
/// // wait for the thread to start up
/// let &(ref lock, ref cvar) = &*pair;
/// let mut started = lock.lock().unwrap();
/// while !*started {
///     started = cvar.wait(started).unwrap();
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Condvar { inner: Box<StaticCondvar> }

/// Statically allocated condition variables.
///
/// This structure is identical to `Condvar` except that it is suitable for use
/// in static initializers for other structures.
///
/// # Examples
///
/// ```
/// #![feature(static_condvar)]
///
/// use std::sync::{StaticCondvar, CONDVAR_INIT};
///
/// static CVAR: StaticCondvar = CONDVAR_INIT;
/// ```
#[unstable(feature = "static_condvar",
           reason = "may be merged with Condvar in the future",
           issue = "27717")]
pub struct StaticCondvar {
    inner: sys::Condvar,
    mutex: AtomicUsize,
}

/// Constant initializer for a statically allocated condition variable.
#[unstable(feature = "static_condvar",
           reason = "may be merged with Condvar in the future",
           issue = "27717")]
pub const CONDVAR_INIT: StaticCondvar = StaticCondvar::new();

impl Condvar {
    /// Creates a new condition variable which is ready to be waited on and
    /// notified.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Condvar {
        Condvar {
            inner: box StaticCondvar {
                inner: sys::Condvar::new(),
                mutex: AtomicUsize::new(0),
            }
        }
    }

    /// Blocks the current thread until this condition variable receives a
    /// notification.
    ///
    /// This function will atomically unlock the mutex specified (represented by
    /// `mutex_guard`) and block the current thread. This means that any calls
    /// to `notify_*()` which happen logically after the mutex is unlocked are
    /// candidates to wake this thread up. When this function call returns, the
    /// lock specified will have been re-acquired.
    ///
    /// Note that this function is susceptible to spurious wakeups. Condition
    /// variables normally have a boolean predicate associated with them, and
    /// the predicate must always be checked each time this function returns to
    /// protect against spurious wakeups.
    ///
    /// # Failure
    ///
    /// This function will return an error if the mutex being waited on is
    /// poisoned when this thread re-acquires the lock. For more information,
    /// see information about poisoning on the Mutex type.
    ///
    /// # Panics
    ///
    /// This function will `panic!()` if it is used with more than one mutex
    /// over time. Each condition variable is dynamically bound to exactly one
    /// mutex to ensure defined behavior across platforms. If this functionality
    /// is not desired, then unsafe primitives in `sys` are provided.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>)
                       -> LockResult<MutexGuard<'a, T>> {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait(guard)
        }
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to `wait()`
    /// except that the thread will be blocked for roughly no longer
    /// than `ms` milliseconds. This method should not be used for
    /// precise timing due to anomalies such as preemption or platform
    /// differences that may not cause the maximum amount of time
    /// waited to be precisely `ms`.
    ///
    /// The returned boolean is `false` only if the timeout is known
    /// to have elapsed.
    ///
    /// Like `wait`, the lock specified will be re-acquired when this function
    /// returns, regardless of whether the timeout elapsed or not.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.6.0", reason = "replaced by `std::sync::Condvar::wait_timeout`")]
    #[allow(deprecated)]
    pub fn wait_timeout_ms<'a, T>(&self, guard: MutexGuard<'a, T>, ms: u32)
                                  -> LockResult<(MutexGuard<'a, T>, bool)> {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait_timeout_ms(guard, ms)
        }
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to `wait()` except that
    /// the thread will be blocked for roughly no longer than `dur`. This
    /// method should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that may not cause the maximum
    /// amount of time waited to be precisely `dur`.
    ///
    /// The returned `WaitTimeoutResult` value indicates if the timeout is
    /// known to have elapsed.
    ///
    /// Like `wait`, the lock specified will be re-acquired when this function
    /// returns, regardless of whether the timeout elapsed or not.
    #[stable(feature = "wait_timeout", since = "1.5.0")]
    pub fn wait_timeout<'a, T>(&self, guard: MutexGuard<'a, T>,
                               dur: Duration)
                               -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)> {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait_timeout(guard, dur)
        }
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to `wait_timeout` except
    /// that the implementation will repeatedly wait while the duration has not
    /// passed and the provided function returns `false`.
    #[unstable(feature = "wait_timeout_with",
               reason = "unsure if this API is broadly needed or what form it should take",
               issue = "27748")]
    pub fn wait_timeout_with<'a, T, F>(&self,
                                       guard: MutexGuard<'a, T>,
                                       dur: Duration,
                                       f: F)
                                       -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)>
            where F: FnMut(LockResult<&mut T>) -> bool {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait_timeout_with(guard, dur, f)
        }
    }

    /// Wakes up one blocked thread on this condvar.
    ///
    /// If there is a blocked thread on this condition variable, then it will
    /// be woken up from its call to `wait` or `wait_timeout`. Calls to
    /// `notify_one` are not buffered in any way.
    ///
    /// To wake up all threads, see `notify_all()`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn notify_one(&self) { unsafe { self.inner.inner.notify_one() } }

    /// Wakes up all blocked threads on this condvar.
    ///
    /// This method will ensure that any current waiters on the condition
    /// variable are awoken. Calls to `notify_all()` are not buffered in any
    /// way.
    ///
    /// To wake up only one thread, see `notify_one()`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn notify_all(&self) { unsafe { self.inner.inner.notify_all() } }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Drop for Condvar {
    fn drop(&mut self) {
        unsafe { self.inner.inner.destroy() }
    }
}

impl StaticCondvar {
    /// Creates a new condition variable
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub const fn new() -> StaticCondvar {
        StaticCondvar {
            inner: sys::Condvar::new(),
            mutex: AtomicUsize::new(0),
        }
    }

    /// Blocks the current thread until this condition variable receives a
    /// notification.
    ///
    /// See `Condvar::wait`.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub fn wait<'a, T>(&'static self, guard: MutexGuard<'a, T>)
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

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// See `Condvar::wait_timeout`.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    #[deprecated(since = "1.6.0", reason = "replaced by `std::sync::StaticCondvar::wait_timeout`")]
    pub fn wait_timeout_ms<'a, T>(&'static self, guard: MutexGuard<'a, T>, ms: u32)
                                  -> LockResult<(MutexGuard<'a, T>, bool)> {
        match self.wait_timeout(guard, Duration::from_millis(ms as u64)) {
            Ok((guard, timed_out)) => Ok((guard, !timed_out.timed_out())),
            Err(poison) => {
                let (guard, timed_out) = poison.into_inner();
                Err(PoisonError::new((guard, !timed_out.timed_out())))
            }
        }
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// See `Condvar::wait_timeout`.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub fn wait_timeout<'a, T>(&'static self,
                               guard: MutexGuard<'a, T>,
                               timeout: Duration)
                               -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)> {
        let (poisoned, result) = unsafe {
            let lock = mutex::guard_lock(&guard);
            self.verify(lock);
            let success = self.inner.wait_timeout(lock, timeout);
            (mutex::guard_poison(&guard).get(), WaitTimeoutResult(!success))
        };
        if poisoned {
            Err(PoisonError::new((guard, result)))
        } else {
            Ok((guard, result))
        }
    }

    /// Waits on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The implementation will repeatedly wait while the duration has not
    /// passed and the function returns `false`.
    ///
    /// See `Condvar::wait_timeout_with`.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub fn wait_timeout_with<'a, T, F>(&'static self,
                                       guard: MutexGuard<'a, T>,
                                       dur: Duration,
                                       mut f: F)
                                       -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)>
            where F: FnMut(LockResult<&mut T>) -> bool {
        // This could be made more efficient by pushing the implementation into
        // sys::condvar
        let start = Instant::now();
        let mut guard_result: LockResult<MutexGuard<'a, T>> = Ok(guard);
        while !f(guard_result
                    .as_mut()
                    .map(|g| &mut **g)
                    .map_err(|e| PoisonError::new(&mut **e.get_mut()))) {
            let consumed = start.elapsed();
            let guard = guard_result.unwrap_or_else(|e| e.into_inner());
            let (new_guard_result, timed_out) = if consumed > dur {
                (Ok(guard), WaitTimeoutResult(true))
            } else {
                match self.wait_timeout(guard, dur - consumed) {
                    Ok((new_guard, timed_out)) => (Ok(new_guard), timed_out),
                    Err(err) => {
                        let (new_guard, no_timeout) = err.into_inner();
                        (Err(PoisonError::new(new_guard)), no_timeout)
                    }
                }
            };
            guard_result = new_guard_result;
            if timed_out.timed_out() {
                let result = f(guard_result
                                    .as_mut()
                                    .map(|g| &mut **g)
                                    .map_err(|e| PoisonError::new(&mut **e.get_mut())));
                let result = WaitTimeoutResult(!result);
                return poison::map_result(guard_result, |g| (g, result));
            }
        }

        poison::map_result(guard_result, |g| (g, WaitTimeoutResult(false)))
    }

    /// Wakes up one blocked thread on this condvar.
    ///
    /// See `Condvar::notify_one`.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub fn notify_one(&'static self) { unsafe { self.inner.notify_one() } }

    /// Wakes up all blocked threads on this condvar.
    ///
    /// See `Condvar::notify_all`.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub fn notify_all(&'static self) { unsafe { self.inner.notify_all() } }

    /// Deallocates all resources associated with this static condvar.
    ///
    /// This method is unsafe to call as there is no guarantee that there are no
    /// active users of the condvar, and this also doesn't prevent any future
    /// users of the condvar. This method is required to be called to not leak
    /// memory on all platforms.
    #[unstable(feature = "static_condvar",
               reason = "may be merged with Condvar in the future",
               issue = "27717")]
    pub unsafe fn destroy(&'static self) {
        self.inner.destroy()
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

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use super::StaticCondvar;
    use sync::mpsc::channel;
    use sync::{StaticMutex, Condvar, Mutex, Arc};
    use sync::atomic::{AtomicUsize, Ordering};
    use thread;
    use time::Duration;
    use u32;

    #[test]
    fn smoke() {
        let c = Condvar::new();
        c.notify_one();
        c.notify_all();
    }

    #[test]
    fn static_smoke() {
        static C: StaticCondvar = StaticCondvar::new();
        C.notify_one();
        C.notify_all();
        unsafe { C.destroy(); }
    }

    #[test]
    fn notify_one() {
        static C: StaticCondvar = StaticCondvar::new();
        static M: StaticMutex = StaticMutex::new();

        let g = M.lock().unwrap();
        let _t = thread::spawn(move|| {
            let _g = M.lock().unwrap();
            C.notify_one();
        });
        let g = C.wait(g).unwrap();
        drop(g);
        unsafe { C.destroy(); M.destroy(); }
    }

    #[test]
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
    fn wait_timeout_ms() {
        static C: StaticCondvar = StaticCondvar::new();
        static M: StaticMutex = StaticMutex::new();

        let g = M.lock().unwrap();
        let (g, _no_timeout) = C.wait_timeout_ms(g, 1).unwrap();
        // spurious wakeups mean this isn't necessarily true
        // assert!(!no_timeout);
        let _t = thread::spawn(move || {
            let _g = M.lock().unwrap();
            C.notify_one();
        });
        let (g, no_timeout) = C.wait_timeout_ms(g, u32::MAX).unwrap();
        assert!(no_timeout);
        drop(g);
        unsafe { C.destroy(); M.destroy(); }
    }

    #[test]
    fn wait_timeout_with() {
        static C: StaticCondvar = StaticCondvar::new();
        static M: StaticMutex = StaticMutex::new();
        static S: AtomicUsize = AtomicUsize::new(0);

        let g = M.lock().unwrap();
        let (g, timed_out) = C.wait_timeout_with(g, Duration::new(0, 1000), |_| {
            false
        }).unwrap();
        assert!(timed_out.timed_out());

        let (tx, rx) = channel();
        let _t = thread::spawn(move || {
            rx.recv().unwrap();
            let g = M.lock().unwrap();
            S.store(1, Ordering::SeqCst);
            C.notify_one();
            drop(g);

            rx.recv().unwrap();
            let g = M.lock().unwrap();
            S.store(2, Ordering::SeqCst);
            C.notify_one();
            drop(g);

            rx.recv().unwrap();
            let _g = M.lock().unwrap();
            S.store(3, Ordering::SeqCst);
            C.notify_one();
        });

        let mut state = 0;
        let day = 24 * 60 * 60;
        let (_g, timed_out) = C.wait_timeout_with(g, Duration::new(day, 0), |_| {
            assert_eq!(state, S.load(Ordering::SeqCst));
            tx.send(()).unwrap();
            state += 1;
            match state {
                1|2 => false,
                _ => true,
            }
        }).unwrap();
        assert!(!timed_out.timed_out());
    }

    #[test]
    #[should_panic]
    fn two_mutexes() {
        static M1: StaticMutex = StaticMutex::new();
        static M2: StaticMutex = StaticMutex::new();
        static C: StaticCondvar = StaticCondvar::new();

        let mut g = M1.lock().unwrap();
        let _t = thread::spawn(move|| {
            let _g = M1.lock().unwrap();
            C.notify_one();
        });
        g = C.wait(g).unwrap();
        drop(g);

        let _ = C.wait(M2.lock().unwrap()).unwrap();
    }
}
