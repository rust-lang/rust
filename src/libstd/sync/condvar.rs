// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;

use sync::atomic::{mod, AtomicUint};
use sync::{mutex, StaticMutexGuard};
use sys_common::condvar as sys;
use sys_common::mutex as sys_mutex;
use time::Duration;

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
///     let mut started = lock.lock();
///     *started = true;
///     cvar.notify_one();
/// }).detach();
///
/// // wait for the thread to start up
/// let &(ref lock, ref cvar) = &*pair;
/// let started = lock.lock();
/// while !*started {
///     cvar.wait(&started);
/// }
/// ```
pub struct Condvar { inner: Box<StaticCondvar> }

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
pub struct StaticCondvar {
    inner: sys::Condvar,
    mutex: AtomicUint,
}

/// Constant initializer for a statically allocated condition variable.
pub const CONDVAR_INIT: StaticCondvar = StaticCondvar {
    inner: sys::CONDVAR_INIT,
    mutex: atomic::INIT_ATOMIC_UINT,
};

/// A trait for vaules which can be passed to the waiting methods of condition
/// variables. This is implemented by the mutex guards in this module.
///
/// Note that this trait should likely not be implemented manually unless you
/// really know what you're doing.
pub trait AsMutexGuard {
    #[allow(missing_docs)]
    unsafe fn as_mutex_guard(&self) -> &StaticMutexGuard;
}

impl Condvar {
    /// Creates a new condition variable which is ready to be waited on and
    /// notified.
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
    /// `guard`) and block the current thread. This means that any calls to
    /// `notify_*()` which happen logically after the mutex is unlocked are
    /// candidates to wake this thread up. When this function call returns, the
    /// lock specified will have been re-acquired.
    ///
    /// Note that this function is susceptible to spurious wakeups. Condition
    /// variables normally have a boolean predicate associated with them, and
    /// the predicate must always be checked each time this function returns to
    /// protect against spurious wakeups.
    ///
    /// # Panics
    ///
    /// This function will `panic!()` if it is used with more than one mutex
    /// over time. Each condition variable is dynamically bound to exactly one
    /// mutex to ensure defined behavior across platforms. If this functionality
    /// is not desired, then unsafe primitives in `sys` are provided.
    pub fn wait<T: AsMutexGuard>(&self, mutex_guard: &T) {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait(mutex_guard)
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
    fn wait_timeout<T: AsMutexGuard>(&self, mutex_guard: &T,
                                     dur: Duration) -> bool {
        unsafe {
            let me: &'static Condvar = &*(self as *const _);
            me.inner.wait_timeout(mutex_guard, dur)
        }
    }

    /// Wake up one blocked thread on this condvar.
    ///
    /// If there is a blocked thread on this condition variable, then it will
    /// be woken up from its call to `wait` or `wait_timeout`. Calls to
    /// `notify_one` are not buffered in any way.
    ///
    /// To wake up all threads, see `notify_one()`.
    pub fn notify_one(&self) { unsafe { self.inner.inner.notify_one() } }

    /// Wake up all blocked threads on this condvar.
    ///
    /// This method will ensure that any current waiters on the condition
    /// variable are awoken. Calls to `notify_all()` are not buffered in any
    /// way.
    ///
    /// To wake up only one thread, see `notify_one()`.
    pub fn notify_all(&self) { unsafe { self.inner.inner.notify_all() } }
}

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
    pub fn wait<T: AsMutexGuard>(&'static self, mutex_guard: &T) {
        unsafe {
            let lock = mutex_guard.as_mutex_guard();
            let sys = mutex::guard_lock(lock);
            self.verify(sys);
            self.inner.wait(sys);
            (*mutex::guard_poison(lock)).check("mutex");
        }
    }

    /// Wait on this condition variable for a notification, timing out after a
    /// specified duration.
    ///
    /// See `Condvar::wait_timeout`.
    #[allow(dead_code)] // may want to stabilize this later, see wait_timeout above
    fn wait_timeout<T: AsMutexGuard>(&'static self, mutex_guard: &T,
                                     dur: Duration) -> bool {
        unsafe {
            let lock = mutex_guard.as_mutex_guard();
            let sys = mutex::guard_lock(lock);
            self.verify(sys);
            let ret = self.inner.wait_timeout(sys, dur);
            (*mutex::guard_poison(lock)).check("mutex");
            return ret;
        }
    }

    /// Wake up one blocked thread on this condvar.
    ///
    /// See `Condvar::notify_one`.
    pub fn notify_one(&'static self) { unsafe { self.inner.notify_one() } }

    /// Wake up all blocked threads on this condvar.
    ///
    /// See `Condvar::notify_all`.
    pub fn notify_all(&'static self) { unsafe { self.inner.notify_all() } }

    /// Deallocate all resources associated with this static condvar.
    ///
    /// This method is unsafe to call as there is no guarantee that there are no
    /// active users of the condvar, and this also doesn't prevent any future
    /// users of the condvar. This method is required to be called to not leak
    /// memory on all platforms.
    pub unsafe fn destroy(&'static self) {
        self.inner.destroy()
    }

    fn verify(&self, mutex: &sys_mutex::Mutex) {
        let addr = mutex as *const _ as uint;
        match self.mutex.compare_and_swap(0, addr, atomic::SeqCst) {
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
    use prelude::*;

    use time::Duration;
    use super::{StaticCondvar, CONDVAR_INIT};
    use sync::{StaticMutex, MUTEX_INIT, Condvar, Mutex, Arc};

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

        let g = M.lock();
        spawn(move|| {
            let _g = M.lock();
            C.notify_one();
        });
        C.wait(&g);
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
            spawn(move|| {
                let &(ref lock, ref cond) = &*data;
                let mut cnt = lock.lock();
                *cnt += 1;
                if *cnt == N {
                    tx.send(());
                }
                while *cnt != 0 {
                    cond.wait(&cnt);
                }
                tx.send(());
            });
        }
        drop(tx);

        let &(ref lock, ref cond) = &*data;
        rx.recv();
        let mut cnt = lock.lock();
        *cnt = 0;
        cond.notify_all();
        drop(cnt);

        for _ in range(0, N) {
            rx.recv();
        }
    }

    #[test]
    fn wait_timeout() {
        static C: StaticCondvar = CONDVAR_INIT;
        static M: StaticMutex = MUTEX_INIT;

        let g = M.lock();
        assert!(!C.wait_timeout(&g, Duration::nanoseconds(1000)));
        spawn(move|| {
            let _g = M.lock();
            C.notify_one();
        });
        assert!(C.wait_timeout(&g, Duration::days(1)));
        drop(g);
        unsafe { C.destroy(); M.destroy(); }
    }

    #[test]
    #[should_fail]
    fn two_mutexes() {
        static M1: StaticMutex = MUTEX_INIT;
        static M2: StaticMutex = MUTEX_INIT;
        static C: StaticCondvar = CONDVAR_INIT;

        let g = M1.lock();
        spawn(move|| {
            let _g = M1.lock();
            C.notify_one();
        });
        C.wait(&g);
        drop(g);

        C.wait(&M2.lock());

    }
}
