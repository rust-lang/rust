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

use cell::UnsafeCell;
use kinds::marker;
use sync::{poison, AsMutexGuard};
use sys_common::mutex as sys;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can also be statically initialized or created via a `new`
/// constructor. Each mutex has a type parameter which represents the data that
/// it is protecting. The data can only be accessed through the RAII guards
/// returned from `lock` and `try_lock`, which guarantees that the data is only
/// ever accessed when the mutex is locked.
///
/// # Poisoning
///
/// In order to prevent access to otherwise invalid data, each mutex will
/// propagate any panics which occur while the lock is held. Once a thread has
/// panicked while holding the lock, then all other threads will immediately
/// panic as well once they hold the lock.
///
/// # Example
///
/// ```rust
/// use std::sync::{Arc, Mutex};
/// use std::thread::Thread;
/// const N: uint = 10;
///
/// // Spawn a few threads to increment a shared variable (non-atomically), and
/// // let the main thread know once all increments are done.
/// //
/// // Here we're using an Arc to share memory among tasks, and the data inside
/// // the Arc is protected with a mutex.
/// let data = Arc::new(Mutex::new(0));
///
/// let (tx, rx) = channel();
/// for _ in range(0u, 10) {
///     let (data, tx) = (data.clone(), tx.clone());
///     Thread::spawn(move|| {
///         // The shared static can only be accessed once the lock is held.
///         // Our non-atomic increment is safe because we're the only thread
///         // which can access the shared state when the lock is held.
///         let mut data = data.lock();
///         *data += 1;
///         if *data == N {
///             tx.send(());
///         }
///         // the lock is unlocked here when `data` goes out of scope.
///     }).detach();
/// }
///
/// rx.recv();
/// ```
pub struct Mutex<T> {
    // Note that this static mutex is in a *box*, not inlined into the struct
    // itself. Once a native mutex has been used once, its address can never
    // change (it can't be moved). This mutex type can be safely moved at any
    // time, so to ensure that the native mutex is used correctly we box the
    // inner lock to give it a constant address.
    inner: Box<StaticMutex>,
    data: UnsafeCell<T>,
}

/// The static mutex type is provided to allow for static allocation of mutexes.
///
/// Note that this is a separate type because using a Mutex correctly means that
/// it needs to have a destructor run. In Rust, statics are not allowed to have
/// destructors. As a result, a `StaticMutex` has one extra method when compared
/// to a `Mutex`, a `destroy` method. This method is unsafe to call, and
/// documentation can be found directly on the method.
///
/// # Example
///
/// ```rust
/// use std::sync::{StaticMutex, MUTEX_INIT};
///
/// static LOCK: StaticMutex = MUTEX_INIT;
///
/// {
///     let _g = LOCK.lock();
///     // do some productive work
/// }
/// // lock is unlocked here.
/// ```
pub struct StaticMutex {
    lock: sys::Mutex,
    poison: UnsafeCell<poison::Flag>,
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be access through this guard via its
/// Deref and DerefMut implementations
#[must_use]
pub struct MutexGuard<'a, T: 'a> {
    // funny underscores due to how Deref/DerefMut currently work (they
    // disregard field privacy).
    __lock: &'a Mutex<T>,
    __guard: StaticMutexGuard,
}

/// An RAII implementation of a "scoped lock" of a static mutex. When this
/// structure is dropped (falls out of scope), the lock will be unlocked.
#[must_use]
pub struct StaticMutexGuard {
    lock: &'static sys::Mutex,
    marker: marker::NoSend,
    poison: poison::Guard<'static>,
}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
pub const MUTEX_INIT: StaticMutex = StaticMutex {
    lock: sys::MUTEX_INIT,
    poison: UnsafeCell { value: poison::Flag { failed: false } },
};

impl<T: Send> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    pub fn new(t: T) -> Mutex<T> {
        Mutex {
            inner: box MUTEX_INIT,
            data: UnsafeCell::new(t),
        }
    }

    /// Acquires a mutex, blocking the current task until it is able to do so.
    ///
    /// This function will block the local task until it is available to acquire
    /// the mutex. Upon returning, the task is the only task with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    ///
    /// # Panics
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will immediately panic once the mutex is acquired.
    pub fn lock(&self) -> MutexGuard<T> {
        unsafe {
            let lock: &'static StaticMutex = &*(&*self.inner as *const _);
            MutexGuard::new(self, lock.lock())
        }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    ///
    /// # Panics
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will immediately panic if the mutex would otherwise be
    /// acquired.
    pub fn try_lock(&self) -> Option<MutexGuard<T>> {
        unsafe {
            let lock: &'static StaticMutex = &*(&*self.inner as *const _);
            lock.try_lock().map(|guard| {
                MutexGuard::new(self, guard)
            })
        }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Mutex<T> {
    fn drop(&mut self) {
        // This is actually safe b/c we know that there is no further usage of
        // this mutex (it's up to the user to arrange for a mutex to get
        // dropped, that's not our job)
        unsafe { self.inner.lock.destroy() }
    }
}

impl StaticMutex {
    /// Acquires this lock, see `Mutex::lock`
    pub fn lock(&'static self) -> StaticMutexGuard {
        unsafe { self.lock.lock() }
        StaticMutexGuard::new(self)
    }

    /// Attempts to grab this lock, see `Mutex::try_lock`
    pub fn try_lock(&'static self) -> Option<StaticMutexGuard> {
        if unsafe { self.lock.try_lock() } {
            Some(StaticMutexGuard::new(self))
        } else {
            None
        }
    }

    /// Deallocates resources associated with this static mutex.
    ///
    /// This method is unsafe because it provides no guarantees that there are
    /// no active users of this mutex, and safety is not guaranteed if there are
    /// active users of this mutex.
    ///
    /// This method is required to ensure that there are no memory leaks on
    /// *all* platforms. It may be the case that some platforms do not leak
    /// memory if this method is not called, but this is not guaranteed to be
    /// true on all platforms.
    pub unsafe fn destroy(&'static self) {
        self.lock.destroy()
    }
}

impl<'mutex, T> MutexGuard<'mutex, T> {
    fn new(lock: &Mutex<T>, guard: StaticMutexGuard) -> MutexGuard<T> {
        MutexGuard { __lock: lock, __guard: guard }
    }
}

impl<'mutex, T> AsMutexGuard for MutexGuard<'mutex, T> {
    unsafe fn as_mutex_guard(&self) -> &StaticMutexGuard { &self.__guard }
}

impl<'mutex, T> Deref<T> for MutexGuard<'mutex, T> {
    fn deref<'a>(&'a self) -> &'a T { unsafe { &*self.__lock.data.get() } }
}
impl<'mutex, T> DerefMut<T> for MutexGuard<'mutex, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.__lock.data.get() }
    }
}

impl StaticMutexGuard {
    fn new(lock: &'static StaticMutex) -> StaticMutexGuard {
        unsafe {
            let guard = StaticMutexGuard {
                lock: &lock.lock,
                marker: marker::NoSend,
                poison: (*lock.poison.get()).borrow(),
            };
            guard.poison.check("mutex");
            return guard;
        }
    }
}

pub fn guard_lock(guard: &StaticMutexGuard) -> &sys::Mutex { guard.lock }
pub fn guard_poison(guard: &StaticMutexGuard) -> &poison::Guard {
    &guard.poison
}

impl AsMutexGuard for StaticMutexGuard {
    unsafe fn as_mutex_guard(&self) -> &StaticMutexGuard { self }
}

#[unsafe_destructor]
impl Drop for StaticMutexGuard {
    fn drop(&mut self) {
        unsafe {
            self.poison.done();
            self.lock.unlock();
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use thread::Thread;
    use sync::{Arc, Mutex, StaticMutex, MUTEX_INIT, Condvar};

    #[test]
    fn smoke() {
        let m = Mutex::new(());
        drop(m.lock());
        drop(m.lock());
    }

    #[test]
    fn smoke_static() {
        static M: StaticMutex = MUTEX_INIT;
        unsafe {
            drop(M.lock());
            drop(M.lock());
            M.destroy();
        }
    }

    #[test]
    fn lots_and_lots() {
        static M: StaticMutex = MUTEX_INIT;
        static mut CNT: uint = 0;
        static J: uint = 1000;
        static K: uint = 3;

        fn inc() {
            for _ in range(0, J) {
                unsafe {
                    let _g = M.lock();
                    CNT += 1;
                }
            }
        }

        let (tx, rx) = channel();
        for _ in range(0, K) {
            let tx2 = tx.clone();
            spawn(move|| { inc(); tx2.send(()); });
            let tx2 = tx.clone();
            spawn(move|| { inc(); tx2.send(()); });
        }

        drop(tx);
        for _ in range(0, 2 * K) {
            rx.recv();
        }
        assert_eq!(unsafe {CNT}, J * K * 2);
        unsafe {
            M.destroy();
        }
    }

    #[test]
    fn try_lock() {
        let m = Mutex::new(());
        assert!(m.try_lock().is_some());
    }

    #[test]
    fn test_mutex_arc_condvar() {
        let arc = Arc::new((Mutex::new(false), Condvar::new()));
        let arc2 = arc.clone();
        let (tx, rx) = channel();
        spawn(move|| {
            // wait until parent gets in
            rx.recv();
            let &(ref lock, ref cvar) = &*arc2;
            let mut lock = lock.lock();
            *lock = true;
            cvar.notify_one();
        });

        let &(ref lock, ref cvar) = &*arc;
        let lock = lock.lock();
        tx.send(());
        assert!(!*lock);
        while !*lock {
            cvar.wait(&lock);
        }
    }

    #[test]
    #[should_fail]
    fn test_arc_condvar_poison() {
        let arc = Arc::new((Mutex::new(1i), Condvar::new()));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        spawn(move|| {
            rx.recv();
            let &(ref lock, ref cvar) = &*arc2;
            let _g = lock.lock();
            cvar.notify_one();
            // Parent should fail when it wakes up.
            panic!();
        });

        let &(ref lock, ref cvar) = &*arc;
        let lock = lock.lock();
        tx.send(());
        while *lock == 1 {
            cvar.wait(&lock);
        }
    }

    #[test]
    #[should_fail]
    fn test_mutex_arc_poison() {
        let arc = Arc::new(Mutex::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| {
            let lock = arc2.lock();
            assert_eq!(*lock, 2);
        }).join();
        let lock = arc.lock();
        assert_eq!(*lock, 1);
    }

    #[test]
    fn test_mutex_arc_nested() {
        // Tests nested mutexes and access
        // to underlying data.
        let arc = Arc::new(Mutex::new(1i));
        let arc2 = Arc::new(Mutex::new(arc));
        let (tx, rx) = channel();
        spawn(move|| {
            let lock = arc2.lock();
            let lock2 = lock.deref().lock();
            assert_eq!(*lock2, 1);
            tx.send(());
        });
        rx.recv();
    }

    #[test]
    fn test_mutex_arc_access_in_unwind() {
        let arc = Arc::new(Mutex::new(1i));
        let arc2 = arc.clone();
        let _ = Thread::spawn(move|| -> () {
            struct Unwinder {
                i: Arc<Mutex<int>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    *self.i.lock() += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        }).join();
        let lock = arc.lock();
        assert_eq!(*lock, 2);
    }
}
