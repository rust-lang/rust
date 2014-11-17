// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A simple native mutex implementation. Warning: this API is likely
//! to change soon.

#![allow(dead_code)]

use core::prelude::*;
use alloc::boxed::Box;
use rustrt::mutex;

pub const LOCKED: uint = 1 << 0;
pub const BLOCKED: uint = 1 << 1;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will properly block tasks waiting for the lock to become
/// available. The mutex can also be statically initialized or created via a
/// `new` constructor.
///
/// # Example
///
/// ```rust
/// use sync::mutex::Mutex;
///
/// let m = Mutex::new();
/// let guard = m.lock();
/// // do some work
/// drop(guard); // unlock the lock
/// ```
pub struct Mutex {
    // Note that this static mutex is in a *box*, not inlined into the struct
    // itself. This is done for memory safety reasons with the usage of a
    // StaticNativeMutex inside the static mutex above. Once a native mutex has
    // been used once, its address can never change (it can't be moved). This
    // mutex type can be safely moved at any time, so to ensure that the native
    // mutex is used correctly we box the inner lock to give it a constant
    // address.
    lock: Box<StaticMutex>,
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
/// use sync::mutex::{StaticMutex, MUTEX_INIT};
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
    lock: mutex::StaticNativeMutex,
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
#[must_use]
pub struct Guard<'a> {
    guard: mutex::LockGuard<'a>,
}

fn lift_guard(guard: mutex::LockGuard) -> Guard {
    Guard { guard: guard }
}

/// Static initialization of a mutex. This constant can be used to initialize
/// other mutex constants.
pub const MUTEX_INIT: StaticMutex = StaticMutex {
    lock: mutex::NATIVE_MUTEX_INIT
};

impl StaticMutex {
    /// Attempts to grab this lock, see `Mutex::try_lock`
    pub fn try_lock<'a>(&'a self) -> Option<Guard<'a>> {
        unsafe { self.lock.trylock().map(lift_guard) }
    }

    /// Acquires this lock, see `Mutex::lock`
    pub fn lock<'a>(&'a self) -> Guard<'a> {
        lift_guard(unsafe { self.lock.lock() })
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
    pub unsafe fn destroy(&self) {
        self.lock.destroy()
    }
}

impl Mutex {
    /// Creates a new mutex in an unlocked state ready for use.
    pub fn new() -> Mutex {
        Mutex {
            lock: box StaticMutex {
                lock: unsafe { mutex::StaticNativeMutex::new() },
            }
        }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    pub fn try_lock<'a>(&'a self) -> Option<Guard<'a>> {
        self.lock.try_lock()
    }

    /// Acquires a mutex, blocking the current task until it is able to do so.
    ///
    /// This function will block the local task until it is available to acquire
    /// the mutex. Upon returning, the task is the only task with the mutex
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the mutex will be unlocked.
    pub fn lock<'a>(&'a self) -> Guard<'a> { self.lock.lock() }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        // This is actually safe b/c we know that there is no further usage of
        // this mutex (it's up to the user to arrange for a mutex to get
        // dropped, that's not our job)
        unsafe { self.lock.destroy() }
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use super::{Mutex, StaticMutex, MUTEX_INIT};

    #[test]
    fn smoke() {
        let m = Mutex::new();
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
            spawn(proc() { inc(); tx2.send(()); });
            let tx2 = tx.clone();
            spawn(proc() { inc(); tx2.send(()); });
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
    fn trylock() {
        let m = Mutex::new();
        assert!(m.try_lock().is_some());
    }
}
