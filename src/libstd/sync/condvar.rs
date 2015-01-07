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

use sync::atomic::{AtomicUint, Ordering, ATOMIC_UINT_INIT};
use sync::poison::{self, LockResult};
use sys_common::condvar as sys;
use sys_common::mutex as sys_mutex;
use time::Duration;
use sync::{mutex, MutexGuard};

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
/// # Example
///
/// ```
/// use std::sync::{Arc, Mutex, Condvar};
/// use std::thread::Thread;
///
/// let pair = Arc::new((Mutex::new(false), Condvar::new()));
/// let pair2 = pair.clone();
///
/// // Inside of our lock, spawn a new thread, and then wait for it to start
/// Thread::spawn(move|| {
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
#[stable]
pub struct Condvar { inner: Box<StaticCondvar> }

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

/// Statically allocated condition variables.
///
/// This structure is identical to `Condvar` except that it is suitable for use
/// in static initializers for other structures.
///
/// # Example
///
/// ```
/// use std::sync::{StaticCondvar, CONDVAR_INIT};
///
/// static CVAR: StaticCondvar = CONDVAR_INIT;
/// ```
#[unstable = "may be merged with Condvar in the future"]
pub struct StaticCondvar {
    inner: sys::Condvar,
    mutex: AtomicUint,
}

unsafe impl Send for StaticCondvar {}
unsafe impl Sync for StaticCondvar {}

/// Constant initializer for a statically allocated condition variable.
#[unstable = "may be merged with Condvar in the future"]
pub const CONDVAR_INIT: StaticCondvar = StaticCondvar {
    inner: sys::CONDVAR_INIT,
    mutex: ATOMIC_UINT_INIT,
};

impl Condvar {
    /// Creates a new condition variable which is ready to be waited on and
    /// notified.
    #[stable]
    pub fn new() -> Condvar {
        Condvar {
            inner: box StaticCondvar {
                inner: unsafe { sys::Condvar::new() },
                mutex: AtomicUint::new(0),
            }
        }
    }

    /// Block the current thread until this condition variable receives a
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
    #[stable]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>)
                       -> LockResult<MutexGuard<'a, T>> {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait(guard)
        }
    }

    /// Wait on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// The semantics of this function are equivalent to `wait()` except that
    /// the thread will be blocked for roughly no longer than `dur`. This method
    /// should not be used for precise timing due to anomalies such as
    /// preemption or platform differences that may not cause the maximum amount
    /// of time waited to be precisely `dur`.
    ///
    /// If the wait timed out, then `false` will be returned. Otherwise if a
    /// notification was received then `true` will be returned.
    ///
    /// Like `wait`, the lock specified will be re-acquired when this function
    /// returns, regardless of whether the timeout elapsed or not.
    // Note that this method is *not* public, and this is quite intentional
    // because we're not quite sure about the semantics of relative vs absolute
    // durations or how the timing guarantees play into what the system APIs
    // provide. There are also additional concerns about the unix-specific
    // implementation which may need to be addressed.
    #[allow(dead_code)]
    fn wait_timeout<'a, T>(&self, guard: MutexGuard<'a, T>, dur: Duration)
                           -> LockResult<(MutexGuard<'a, T>, bool)> {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait_timeout(guard, dur)
        }
    }

    /// Wake up one blocked thread on this condvar.
    ///
    /// If there is a blocked thread on this condition variable, then it will
    /// be woken up from its call to `wait` or `wait_timeout`. Calls to
    /// `notify_one` are not buffered in any way.
    ///
    /// To wake up all threads, see `notify_one()`.
    #[stable]
    pub fn notify_one(&self) { unsafe { self.inner.inner.notify_one() } }

    /// Wake up all blocked threads on this condvar.
    ///
    /// This method will ensure that any current waiters on the condition
    /// variable are awoken. Calls to `notify_all()` are not buffered in any
    /// way.
    ///
    /// To wake up only one thread, see `notify_one()`.
    #[stable]
    pub fn notify_all(&self) { unsafe { self.inner.inner.notify_all() } }
}

#[stable]
impl Drop for Condvar {
    fn drop(&mut self) {
        unsafe { self.inner.inner.destroy() }
    }
}

impl StaticCondvar {
    /// Block the current thread until this condition variable receives a
    /// notification.
    ///
    /// See `Condvar::wait`.
    #[unstable = "may be merged with Condvar in the future"]
    pub fn wait<'a, T>(&'static self, guard: MutexGuard<'a, T>)
                       -> LockResult<MutexGuard<'a, T>> {
        let poisoned = unsafe {
            let lock = mutex::guard_lock(&guard);
            self.verify(lock);
            self.inner.wait(lock);
            mutex::guard_poison(&guard).get()
        };
        if poisoned {
            Err(poison::new_poison_error(guard))
        } else {
            Ok(guard)
        }
    }

    /// Wait on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// See `Condvar::wait_timeout`.
    #[allow(dead_code)] // may want to stabilize this later, see wait_timeout above
    fn wait_timeout<'a, T>(&'static self, guard: MutexGuard<'a, T>, dur: Duration)
                           -> LockResult<(MutexGuard<'a, T>, bool)> {
        let (poisoned, success) = unsafe {
            let lock = mutex::guard_lock(&guard);
            self.verify(lock);
            let success = self.inner.wait_timeout(lock, dur);
            (mutex::guard_poison(&guard).get(), success)
        };
        if poisoned {
            Err(poison::new_poison_error((guard, success)))
        } else {
            Ok((guard, success))
        }
    }

    /// Wake up one blocked thread on this condvar.
    ///
    /// See `Condvar::notify_one`.
    #[unstable = "may be merged with Condvar in the future"]
    pub fn notify_one(&'static self) { unsafe { self.inner.notify_one() } }

    /// Wake up all blocked threads on this condvar.
    ///
    /// See `Condvar::notify_all`.
    #[unstable = "may be merged with Condvar in the future"]
    pub fn notify_all(&'static self) { unsafe { self.inner.notify_all() } }

    /// Deallocate all resources associated with this static condvar.
    ///
    /// This method is unsafe to call as there is no guarantee that there are no
    /// active users of the condvar, and this also doesn't prevent any future
    /// users of the condvar. This method is required to be called to not leak
    /// memory on all platforms.
    #[unstable = "may be merged with Condvar in the future"]
    pub unsafe fn destroy(&'static self) {
        self.inner.destroy()
    }

    fn verify(&self, mutex: &sys_mutex::Mutex) {
        let addr = mutex as *const _ as uint;
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

    use super::{StaticCondvar, CONDVAR_INIT};
    use sync::mpsc::channel;
    use sync::{StaticMutex, MUTEX_INIT, Condvar, Mutex, Arc};
    use thread::Thread;
    use time::Duration;

    #[test]
    fn smoke() {
        let c = Condvar::new();
        c.notify_one();
        c.notify_all();
    }

    #[test]
    fn static_smoke() {
        static C: StaticCondvar = CONDVAR_INIT;
        C.notify_one();
        C.notify_all();
        unsafe { C.destroy(); }
    }

    #[test]
    fn notify_one() {
        static C: StaticCondvar = CONDVAR_INIT;
        static M: StaticMutex = MUTEX_INIT;

        let g = M.lock().unwrap();
        let _t = Thread::spawn(move|| {
            let _g = M.lock().unwrap();
            C.notify_one();
        });
        let g = C.wait(g).unwrap();
        drop(g);
        unsafe { C.destroy(); M.destroy(); }
    }

    #[test]
    fn notify_all() {
        const N: uint = 10;

        let data = Arc::new((Mutex::new(0), Condvar::new()));
        let (tx, rx) = channel();
        for _ in range(0, N) {
            let data = data.clone();
            let tx = tx.clone();
            Thread::spawn(move|| {
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

        for _ in range(0, N) {
            rx.recv().unwrap();
        }
    }

    #[test]
    fn wait_timeout() {
        static C: StaticCondvar = CONDVAR_INIT;
        static M: StaticMutex = MUTEX_INIT;

        let g = M.lock().unwrap();
        let (g, success) = C.wait_timeout(g, Duration::nanoseconds(1000)).unwrap();
        assert!(!success);
        let _t = Thread::spawn(move || {
            let _g = M.lock().unwrap();
            C.notify_one();
        });
        let (g, success) = C.wait_timeout(g, Duration::days(1)).unwrap();
        assert!(success);
        drop(g);
        unsafe { C.destroy(); M.destroy(); }
    }

    #[test]
    #[should_fail]
    fn two_mutexes() {
        static M1: StaticMutex = MUTEX_INIT;
        static M2: StaticMutex = MUTEX_INIT;
        static C: StaticCondvar = CONDVAR_INIT;

        let mut g = M1.lock().unwrap();
        let _t = Thread::spawn(move|| {
            let _g = M1.lock().unwrap();
            C.notify_one();
        });
        g = C.wait(g).unwrap();
        drop(g);

        let _ = C.wait(M2.lock().unwrap()).unwrap();
    }
}
