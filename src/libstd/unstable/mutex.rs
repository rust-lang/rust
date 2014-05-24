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
//! use std::unstable::mutex::{NativeMutex, StaticNativeMutex, NATIVE_MUTEX_INIT};
//!
//! // Use a statically initialized mutex
//! static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
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

use option::{Option, None, Some};
use ops::Drop;

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
/// failure, because the lock will be automatically unlocked even
/// then.
#[must_use]
pub struct LockGuard<'a> {
    lock: &'a StaticNativeMutex
}

pub static NATIVE_MUTEX_INIT: StaticNativeMutex = StaticNativeMutex {
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
    /// use std::unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
    /// static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
    /// unsafe {
    ///     let _guard = LOCK.lock();
    ///     // critical section...
    /// } // automatically unlocked in `_guard`'s destructor
    /// ```
    pub unsafe fn lock<'a>(&'a self) -> LockGuard<'a> {
        self.inner.lock();

        LockGuard { lock: self }
    }

    /// Attempts to acquire the lock. The value returned is `Some` if
    /// the attempt succeeded.
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
    pub unsafe fn lock_noguard(&self) { self.inner.lock() }

    /// Attempts to acquire the lock without creating a
    /// `LockGuard`. The value returned is whether the lock was
    /// acquired or not.
    ///
    /// If `true` is returned, this needs to be paired with a call to
    /// `.unlock_noguard`. Prefer using `.trylock`.
    pub unsafe fn trylock_noguard(&self) -> bool {
        self.inner.trylock()
    }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    pub unsafe fn unlock_noguard(&self) { self.inner.unlock() }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held. Prefer
    /// using `LockGuard.wait` since that guarantees that the lock is
    /// held.
    pub unsafe fn wait_noguard(&self) { self.inner.wait() }

    /// Signals a thread in `wait` to wake up
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
    /// ```rust
    /// use std::unstable::mutex::NativeMutex;
    /// unsafe {
    ///     let mut lock = NativeMutex::new();
    ///
    ///     {
    ///         let _guard = lock.lock();
    ///         // critical section...
    ///     } // automatically unlocked in `_guard`'s destructor
    /// }
    /// ```
    pub unsafe fn lock<'a>(&'a self) -> LockGuard<'a> {
        self.inner.lock()
    }

    /// Attempts to acquire the lock. The value returned is `Some` if
    /// the attempt succeeded.
    pub unsafe fn trylock<'a>(&'a self) -> Option<LockGuard<'a>> {
        self.inner.trylock()
    }

    /// Acquire the lock without creating a `LockGuard`.
    ///
    /// These needs to be paired with a call to `.unlock_noguard`. Prefer using
    /// `.lock`.
    pub unsafe fn lock_noguard(&self) { self.inner.lock_noguard() }

    /// Attempts to acquire the lock without creating a
    /// `LockGuard`. The value returned is whether the lock was
    /// acquired or not.
    ///
    /// If `true` is returned, this needs to be paired with a call to
    /// `.unlock_noguard`. Prefer using `.trylock`.
    pub unsafe fn trylock_noguard(&self) -> bool {
        self.inner.trylock_noguard()
    }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    pub unsafe fn unlock_noguard(&self) { self.inner.unlock_noguard() }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held. Prefer
    /// using `LockGuard.wait` since that guarantees that the lock is
    /// held.
    pub unsafe fn wait_noguard(&self) { self.inner.wait_noguard() }

    /// Signals a thread in `wait` to wake up
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

#[cfg(unix)]
mod imp {
    use libc;
    use self::os::{PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER,
                   pthread_mutex_t, pthread_cond_t};
    use ty::Unsafe;
    use kinds::marker;

    type pthread_mutexattr_t = libc::c_void;
    type pthread_condattr_t = libc::c_void;

    #[cfg(target_os = "freebsd")]
    mod os {
        use libc;

        pub type pthread_mutex_t = *libc::c_void;
        pub type pthread_cond_t = *libc::c_void;

        pub static PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t =
            0 as pthread_mutex_t;
        pub static PTHREAD_COND_INITIALIZER: pthread_cond_t =
            0 as pthread_cond_t;
    }

    #[cfg(target_os = "macos")]
    mod os {
        use libc;

        #[cfg(target_arch = "x86_64")]
        static __PTHREAD_MUTEX_SIZE__: uint = 56;
        #[cfg(target_arch = "x86_64")]
        static __PTHREAD_COND_SIZE__: uint = 40;
        #[cfg(target_arch = "x86")]
        static __PTHREAD_MUTEX_SIZE__: uint = 40;
        #[cfg(target_arch = "x86")]
        static __PTHREAD_COND_SIZE__: uint = 24;

        static _PTHREAD_MUTEX_SIG_init: libc::c_long = 0x32AAABA7;
        static _PTHREAD_COND_SIG_init: libc::c_long = 0x3CB0B1BB;

        pub struct pthread_mutex_t {
            __sig: libc::c_long,
            __opaque: [u8, ..__PTHREAD_MUTEX_SIZE__],
        }
        pub struct pthread_cond_t {
            __sig: libc::c_long,
            __opaque: [u8, ..__PTHREAD_COND_SIZE__],
        }

        pub static PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            __sig: _PTHREAD_MUTEX_SIG_init,
            __opaque: [0, ..__PTHREAD_MUTEX_SIZE__],
        };
        pub static PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
            __sig: _PTHREAD_COND_SIG_init,
            __opaque: [0, ..__PTHREAD_COND_SIZE__],
        };
    }

    #[cfg(target_os = "linux")]
    mod os {
        use libc;

        // minus 8 because we have an 'align' field
        #[cfg(target_arch = "x86_64")]
        static __SIZEOF_PTHREAD_MUTEX_T: uint = 40 - 8;
        #[cfg(target_arch = "x86")]
        static __SIZEOF_PTHREAD_MUTEX_T: uint = 24 - 8;
        #[cfg(target_arch = "arm")]
        static __SIZEOF_PTHREAD_MUTEX_T: uint = 24 - 8;
        #[cfg(target_arch = "mips")]
        static __SIZEOF_PTHREAD_MUTEX_T: uint = 24 - 8;
        #[cfg(target_arch = "x86_64")]
        static __SIZEOF_PTHREAD_COND_T: uint = 48 - 8;
        #[cfg(target_arch = "x86")]
        static __SIZEOF_PTHREAD_COND_T: uint = 48 - 8;
        #[cfg(target_arch = "arm")]
        static __SIZEOF_PTHREAD_COND_T: uint = 48 - 8;
        #[cfg(target_arch = "mips")]
        static __SIZEOF_PTHREAD_COND_T: uint = 48 - 8;

        pub struct pthread_mutex_t {
            __align: libc::c_longlong,
            size: [u8, ..__SIZEOF_PTHREAD_MUTEX_T],
        }
        pub struct pthread_cond_t {
            __align: libc::c_longlong,
            size: [u8, ..__SIZEOF_PTHREAD_COND_T],
        }

        pub static PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            __align: 0,
            size: [0, ..__SIZEOF_PTHREAD_MUTEX_T],
        };
        pub static PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
            __align: 0,
            size: [0, ..__SIZEOF_PTHREAD_COND_T],
        };
    }
    #[cfg(target_os = "android")]
    mod os {
        use libc;

        pub struct pthread_mutex_t { value: libc::c_int }
        pub struct pthread_cond_t { value: libc::c_int }

        pub static PTHREAD_MUTEX_INITIALIZER: pthread_mutex_t = pthread_mutex_t {
            value: 0,
        };
        pub static PTHREAD_COND_INITIALIZER: pthread_cond_t = pthread_cond_t {
            value: 0,
        };
    }

    pub struct Mutex {
        lock: Unsafe<pthread_mutex_t>,
        cond: Unsafe<pthread_cond_t>,
    }

    pub static MUTEX_INIT: Mutex = Mutex {
        lock: Unsafe {
            value: PTHREAD_MUTEX_INITIALIZER,
            marker1: marker::InvariantType,
        },
        cond: Unsafe {
            value: PTHREAD_COND_INITIALIZER,
            marker1: marker::InvariantType,
        },
    };

    impl Mutex {
        pub unsafe fn new() -> Mutex {
            // As mutex might be moved and address is changing it
            // is better to avoid initialization of potentially
            // opaque OS data before it landed
            let m = Mutex {
                lock: Unsafe::new(PTHREAD_MUTEX_INITIALIZER),
                cond: Unsafe::new(PTHREAD_COND_INITIALIZER),
            };

            return m;
        }

        pub unsafe fn lock(&self) { pthread_mutex_lock(self.lock.get()); }
        pub unsafe fn unlock(&self) { pthread_mutex_unlock(self.lock.get()); }
        pub unsafe fn signal(&self) { pthread_cond_signal(self.cond.get()); }
        pub unsafe fn wait(&self) {
            pthread_cond_wait(self.cond.get(), self.lock.get());
        }
        pub unsafe fn trylock(&self) -> bool {
            pthread_mutex_trylock(self.lock.get()) == 0
        }
        pub unsafe fn destroy(&self) {
            pthread_mutex_destroy(self.lock.get());
            pthread_cond_destroy(self.cond.get());
        }
    }

    extern {
        fn pthread_mutex_destroy(lock: *mut pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_destroy(cond: *mut pthread_cond_t) -> libc::c_int;
        fn pthread_mutex_lock(lock: *mut pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_trylock(lock: *mut pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_unlock(lock: *mut pthread_mutex_t) -> libc::c_int;

        fn pthread_cond_wait(cond: *mut pthread_cond_t,
                             lock: *mut pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_signal(cond: *mut pthread_cond_t) -> libc::c_int;
    }
}

#[cfg(windows)]
mod imp {
    use rt::libc_heap::malloc_raw;
    use libc::{HANDLE, BOOL, LPSECURITY_ATTRIBUTES, c_void, DWORD, LPCSTR};
    use libc;
    use ptr;
    use sync::atomics;

    type LPCRITICAL_SECTION = *mut c_void;
    static SPIN_COUNT: DWORD = 4000;
    #[cfg(target_arch = "x86")]
    static CRIT_SECTION_SIZE: uint = 24;
    #[cfg(target_arch = "x86_64")]
    static CRIT_SECTION_SIZE: uint = 40;

    pub struct Mutex {
        // pointers for the lock/cond handles, atomically updated
        lock: atomics::AtomicUint,
        cond: atomics::AtomicUint,
    }

    pub static MUTEX_INIT: Mutex = Mutex {
        lock: atomics::INIT_ATOMIC_UINT,
        cond: atomics::INIT_ATOMIC_UINT,
    };

    impl Mutex {
        pub unsafe fn new() -> Mutex {
            Mutex {
                lock: atomics::AtomicUint::new(init_lock()),
                cond: atomics::AtomicUint::new(init_cond()),
            }
        }
        pub unsafe fn lock(&self) {
            EnterCriticalSection(self.getlock() as LPCRITICAL_SECTION)
        }
        pub unsafe fn trylock(&self) -> bool {
            TryEnterCriticalSection(self.getlock() as LPCRITICAL_SECTION) != 0
        }
        pub unsafe fn unlock(&self) {
            LeaveCriticalSection(self.getlock() as LPCRITICAL_SECTION)
        }

        pub unsafe fn wait(&self) {
            self.unlock();
            WaitForSingleObject(self.getcond() as HANDLE, libc::INFINITE);
            self.lock();
        }

        pub unsafe fn signal(&self) {
            assert!(SetEvent(self.getcond() as HANDLE) != 0);
        }

        /// This function is especially unsafe because there are no guarantees made
        /// that no other thread is currently holding the lock or waiting on the
        /// condition variable contained inside.
        pub unsafe fn destroy(&self) {
            let lock = self.lock.swap(0, atomics::SeqCst);
            let cond = self.cond.swap(0, atomics::SeqCst);
            if lock != 0 { free_lock(lock) }
            if cond != 0 { free_cond(cond) }
        }

        unsafe fn getlock(&self) -> *mut c_void {
            match self.lock.load(atomics::SeqCst) {
                0 => {}
                n => return n as *mut c_void
            }
            let lock = init_lock();
            match self.lock.compare_and_swap(0, lock, atomics::SeqCst) {
                0 => return lock as *mut c_void,
                _ => {}
            }
            free_lock(lock);
            return self.lock.load(atomics::SeqCst) as *mut c_void;
        }

        unsafe fn getcond(&self) -> *mut c_void {
            match self.cond.load(atomics::SeqCst) {
                0 => {}
                n => return n as *mut c_void
            }
            let cond = init_cond();
            match self.cond.compare_and_swap(0, cond, atomics::SeqCst) {
                0 => return cond as *mut c_void,
                _ => {}
            }
            free_cond(cond);
            return self.cond.load(atomics::SeqCst) as *mut c_void;
        }
    }

    pub unsafe fn init_lock() -> uint {
        let block = malloc_raw(CRIT_SECTION_SIZE as uint) as *mut c_void;
        InitializeCriticalSectionAndSpinCount(block, SPIN_COUNT);
        return block as uint;
    }

    pub unsafe fn init_cond() -> uint {
        return CreateEventA(ptr::mut_null(), libc::FALSE, libc::FALSE,
                            ptr::null()) as uint;
    }

    pub unsafe fn free_lock(h: uint) {
        DeleteCriticalSection(h as LPCRITICAL_SECTION);
        libc::free(h as *mut c_void);
    }

    pub unsafe fn free_cond(h: uint) {
        let block = h as HANDLE;
        libc::CloseHandle(block);
    }

    extern "system" {
        fn CreateEventA(lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                        bManualReset: BOOL,
                        bInitialState: BOOL,
                        lpName: LPCSTR) -> HANDLE;
        fn InitializeCriticalSectionAndSpinCount(
                        lpCriticalSection: LPCRITICAL_SECTION,
                        dwSpinCount: DWORD) -> BOOL;
        fn DeleteCriticalSection(lpCriticalSection: LPCRITICAL_SECTION);
        fn EnterCriticalSection(lpCriticalSection: LPCRITICAL_SECTION);
        fn LeaveCriticalSection(lpCriticalSection: LPCRITICAL_SECTION);
        fn TryEnterCriticalSection(lpCriticalSection: LPCRITICAL_SECTION) -> BOOL;
        fn SetEvent(hEvent: HANDLE) -> BOOL;
        fn WaitForSingleObject(hHandle: HANDLE, dwMilliseconds: DWORD) -> DWORD;
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
        static mut lock: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            let _guard = lock.lock();
        }
    }

    #[test]
    fn smoke_cond() {
        static mut lock: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            let guard = lock.lock();
            let t = Thread::start(proc() {
                let guard = lock.lock();
                guard.signal();
            });
            guard.wait();
            drop(guard);

            t.join();
        }
    }

    #[test]
    fn smoke_lock_noguard() {
        static mut lock: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            lock.lock_noguard();
            lock.unlock_noguard();
        }
    }

    #[test]
    fn smoke_cond_noguard() {
        static mut lock: StaticNativeMutex = NATIVE_MUTEX_INIT;
        unsafe {
            lock.lock_noguard();
            let t = Thread::start(proc() {
                lock.lock_noguard();
                lock.signal_noguard();
                lock.unlock_noguard();
            });
            lock.wait_noguard();
            lock.unlock_noguard();

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
