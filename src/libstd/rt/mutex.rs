// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A native mutex and condition variable type.
//!
//! This module contains bindings to the platform's native mutex/condition
//! variable primitives. It provides two types: `StaticNativeMutex`, which can
//! be statically initialized via the `NATIVE_MUTEX_INIT` value, and a simple
//! wrapper `NativeMutex` that has a destructor to clean up after itself. These
//! objects serve as both mutexes and condition variables simultaneously.
//!
//! The static lock is lazily initialized, but it can only be unsafely
//! destroyed. A statically initialized lock doesn't necessarily have a time at
//! which it can get deallocated. For this reason, there is no `Drop`
//! implementation of the static mutex, but rather the `destroy()` method must
//! be invoked manually if destruction of the mutex is desired.
//!
//! The non-static `NativeMutex` type does have a destructor, but cannot be
//! statically initialized.
//!
//! It is not recommended to use this type for idiomatic rust use. These types
//! are appropriate where no other options are available, but other rust
//! concurrency primitives should be used before them: the `sync` crate defines
//! `StaticMutex` and `Mutex` types.
//!
//! # Example
//!
//! ```rust
//! use rt::mutex::{NativeMutex, StaticNativeMutex, NATIVE_MUTEX_INIT};
//!
//! // Use a statically initialized mutex
//! static LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
//!
//! unsafe {
//!     let _guard = LOCK.lock();
//! } // automatically unlocked here
//!
//! // Use a normally initialized mutex
//! unsafe {
//!     let mut lock = NativeMutex::new();
//!
//!     {
//!         let _guard = lock.lock();
//!     } // unlocked here
//!
//!     // sometimes the RAII guard isn't appropriate
//!     lock.lock_noguard();
//!     lock.unlock_noguard();
//! } // `lock` is deallocated here
//! ```

#![allow(non_camel_case_types)]

use core::prelude::*;

use sys::mutex as imp;

/// A native mutex suitable for storing in statics (that is, it has
/// the `destroy` method rather than a destructor).
///
/// Prefer the `NativeMutex` type where possible, since that does not
/// require manual deallocation.
pub struct StaticNativeMutex {
    inner: imp::Mutex,
}

/// A native mutex with a destructor for clean-up.
///
/// See `StaticNativeMutex` for a version that is suitable for storing in
/// statics.
pub struct NativeMutex {
    inner: StaticNativeMutex
}

/// Automatically unlocks the mutex that it was created from on
/// destruction.
///
/// Using this makes lock-based code resilient to unwinding/task
/// panic, because the lock will be automatically unlocked even
/// then.
#[must_use]
pub struct LockGuard<'a> {
    lock: &'a StaticNativeMutex
}

pub const NATIVE_MUTEX_INIT: StaticNativeMutex = StaticNativeMutex {
    inner: imp::MUTEX_INIT,
};

impl StaticNativeMutex {
    /// Creates a new mutex.
    ///
    /// Note that a mutex created in this way needs to be explicit
    /// freed with a call to `destroy` or it will leak.
    /// Also it is important to avoid locking until mutex has stopped moving
    pub unsafe fn new() -> StaticNativeMutex {
        StaticNativeMutex { inner: imp::Mutex::new() }
    }

    /// Acquires this lock. This assumes that the current thread does not
    /// already hold the lock.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
    /// static LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
    /// unsafe {
    ///     let _guard = LOCK.lock();
    ///     // critical section...
    /// } // automatically unlocked in `_guard`'s destructor
    /// ```
    ///
    /// # Unsafety
    ///
    /// This method is unsafe because it will not function correctly if this
    /// mutex has been *moved* since it was last used. The mutex can move an
    /// arbitrary number of times before its first usage, but once a mutex has
    /// been used once it is no longer allowed to move (or otherwise it invokes
    /// undefined behavior).
    ///
    /// Additionally, this type does not take into account any form of
    /// scheduling model. This will unconditionally block the *os thread* which
    /// is not always desired.
    pub unsafe fn lock<'a>(&'a self) -> LockGuard<'a> {
        self.inner.lock();

        LockGuard { lock: self }
    }

    /// Attempts to acquire the lock. The value returned is `Some` if
    /// the attempt succeeded.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe for the same reasons as `lock`.
    pub unsafe fn trylock<'a>(&'a self) -> Option<LockGuard<'a>> {
        if self.inner.trylock() {
            Some(LockGuard { lock: self })
        } else {
            None
        }
    }

    /// Acquire the lock without creating a `LockGuard`.
    ///
    /// These needs to be paired with a call to `.unlock_noguard`. Prefer using
    /// `.lock`.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe for the same reasons as `lock`. Additionally, this
    /// does not guarantee that the mutex will ever be unlocked, and it is
    /// undefined to drop an already-locked mutex.
    pub unsafe fn lock_noguard(&self) { self.inner.lock() }

    /// Attempts to acquire the lock without creating a
    /// `LockGuard`. The value returned is whether the lock was
    /// acquired or not.
    ///
    /// If `true` is returned, this needs to be paired with a call to
    /// `.unlock_noguard`. Prefer using `.trylock`.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe for the same reasons as `lock_noguard`.
    pub unsafe fn trylock_noguard(&self) -> bool {
        self.inner.trylock()
    }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe for the same reasons as `lock`. Additionally, it
    /// is not guaranteed that this is unlocking a previously locked mutex. It
    /// is undefined to unlock an unlocked mutex.
    pub unsafe fn unlock_noguard(&self) { self.inner.unlock() }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held. Prefer
    /// using `LockGuard.wait` since that guarantees that the lock is
    /// held.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe for the same reasons as `lock`. Additionally, this
    /// is unsafe because the mutex may not be currently locked.
    pub unsafe fn wait_noguard(&self) { self.inner.wait() }

    /// Signals a thread in `wait` to wake up
    ///
    /// # Unsafety
    ///
    /// This method is unsafe for the same reasons as `lock`. Additionally, this
    /// is unsafe because the mutex may not be currently locked.
    pub unsafe fn signal_noguard(&self) { self.inner.signal() }

    /// This function is especially unsafe because there are no guarantees made
    /// that no other thread is currently holding the lock or waiting on the
    /// condition variable contained inside.
    pub unsafe fn destroy(&self) { self.inner.destroy() }
}

impl NativeMutex {
    /// Creates a new mutex.
    ///
    /// The user must be careful to ensure the mutex is not locked when its is
    /// being destroyed.
    /// Also it is important to avoid locking until mutex has stopped moving
    pub unsafe fn new() -> NativeMutex {
        NativeMutex { inner: StaticNativeMutex::new() }
    }

    /// Acquires this lock. This assumes that the current thread does not
    /// already hold the lock.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rt::mutex::NativeMutex;
    /// unsafe {
    ///     let mut lock = NativeMutex::new();
    ///
    ///     {
    ///         let _guard = lock.lock();
    ///         // critical section...
    ///     } // automatically unlocked in `_guard`'s destructor
    /// }
    /// ```
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::lock`.
    pub unsafe fn lock<'a>(&'a self) -> LockGuard<'a> {
        self.inner.lock()
    }

    /// Attempts to acquire the lock. The value returned is `Some` if
    /// the attempt succeeded.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::trylock`.
    pub unsafe fn trylock<'a>(&'a self) -> Option<LockGuard<'a>> {
        self.inner.trylock()
    }

    /// Acquire the lock without creating a `LockGuard`.
    ///
    /// These needs to be paired with a call to `.unlock_noguard`. Prefer using
    /// `.lock`.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::lock_noguard`.
    pub unsafe fn lock_noguard(&self) { self.inner.lock_noguard() }

    /// Attempts to acquire the lock without creating a
    /// `LockGuard`. The value returned is whether the lock was
    /// acquired or not.
    ///
    /// If `true` is returned, this needs to be paired with a call to
    /// `.unlock_noguard`. Prefer using `.trylock`.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::trylock_noguard`.
    pub unsafe fn trylock_noguard(&self) -> bool {
        self.inner.trylock_noguard()
    }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::unlock_noguard`.
    pub unsafe fn unlock_noguard(&self) { self.inner.unlock_noguard() }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held. Prefer
    /// using `LockGuard.wait` since that guarantees that the lock is
    /// held.
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::wait_noguard`.
    pub unsafe fn wait_noguard(&self) { self.inner.wait_noguard() }

    /// Signals a thread in `wait` to wake up
    ///
    /// # Unsafety
    ///
    /// This method is unsafe due to the same reasons as
    /// `StaticNativeMutex::signal_noguard`.
    pub unsafe fn signal_noguard(&self) { self.inner.signal_noguard() }
}

impl Drop for NativeMutex {
    fn drop(&mut self) {
        unsafe {self.inner.destroy()}
    }
}

impl<'a> LockGuard<'a> {
    /// Block on the internal condition variable.
    pub unsafe fn wait(&self) {
        self.lock.wait_noguard()
    }

    /// Signals a thread in `wait` to wake up.
    pub unsafe fn signal(&self) {
        self.lock.signal_noguard()
    }
}

#[unsafe_destructor]
impl<'a> Drop for LockGuard<'a> {
    fn drop(&mut self) {
        unsafe {self.lock.unlock_noguard()}
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use mem::drop;
    use super::{StaticNativeMutex, NATIVE_MUTEX_INIT};
    use rt::thread::Thread;

    #[test]
    fn smoke_lock() {
        static LK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            let _guard = LK.lock();
        }
    }

    #[test]
    fn smoke_cond() {
        static LK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            let guard = LK.lock();
            let t = Thread::start(move|| {
                let guard = LK.lock();
                guard.signal();
            });
            guard.wait();
            drop(guard);

            t.join();
        }
    }

    #[test]
    fn smoke_lock_noguard() {
        static LK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            LK.lock_noguard();
            LK.unlock_noguard();
        }
    }

    #[test]
    fn smoke_cond_noguard() {
        static LK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            LK.lock_noguard();
            let t = Thread::start(move|| {
                LK.lock_noguard();
                LK.signal_noguard();
                LK.unlock_noguard();
            });
            LK.wait_noguard();
            LK.unlock_noguard();

            t.join();
        }
    }

    #[test]
    fn destroy_immediately() {
        unsafe {
            let m = StaticNativeMutex::new();
            m.destroy();
        }
    }
}
