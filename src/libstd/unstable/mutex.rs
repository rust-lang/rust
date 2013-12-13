// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A native mutex and condition variable type
//!
//! This module contains bindings to the platform's native mutex/condition
//! variable primitives. It provides a single type, `Mutex`, which can be
//! statically initialized via the `MUTEX_INIT` value. This object serves as both a
//! mutex and a condition variable simultaneously.
//!
//! The lock is lazily initialized, but it can only be unsafely destroyed. A
//! statically initialized lock doesn't necessarily have a time at which it can
//! get deallocated. For this reason, there is no `Drop` implementation of the
//! mutex, but rather the `destroy()` method must be invoked manually if
//! destruction of the mutex is desired.
//!
//! It is not recommended to use this type for idiomatic rust use. This type is
//! appropriate where no other options are available, but other rust concurrency
//! primitives should be used before this type.
//!
//! # Example
//!
//!     use std::unstable::mutex::{Mutex, MUTEX_INIT};
//!
//!     // Use a statically initialized mutex
//!     static mut lock: Mutex = MUTEX_INIT;
//!
//!     unsafe {
//!         lock.lock();
//!         lock.unlock();
//!     }
//!
//!     // Use a normally initialized mutex
//!     let mut lock = Mutex::new();
//!     unsafe {
//!         lock.lock();
//!         lock.unlock();
//!         lock.destroy();
//!     }

#[allow(non_camel_case_types)];

use libc::c_void;
use sync::atomics;

pub struct Mutex {
    // pointers for the lock/cond handles, atomically updated
    priv lock: atomics::AtomicUint,
    priv cond: atomics::AtomicUint,
}

pub static MUTEX_INIT: Mutex = Mutex {
    lock: atomics::INIT_ATOMIC_UINT,
    cond: atomics::INIT_ATOMIC_UINT,
};

impl Mutex {
    /// Creates a new mutex, with the lock/condition variable pre-initialized
    pub unsafe fn new() -> Mutex {
        Mutex {
            lock: atomics::AtomicUint::new(imp::init_lock() as uint),
            cond: atomics::AtomicUint::new(imp::init_cond() as uint),
        }
    }

    /// Creates a new mutex, with the lock/condition variable not initialized.
    /// This is the same as initializing from the MUTEX_INIT static.
    pub unsafe fn empty() -> Mutex {
        Mutex {
            lock: atomics::AtomicUint::new(0),
            cond: atomics::AtomicUint::new(0),
        }
    }

    /// Creates a new copy of this mutex. This is an unsafe operation because
    /// there is no reference counting performed on this type.
    ///
    /// This function may only be called on mutexes which have had both the
    /// internal condition variable and lock initialized. This means that the
    /// mutex must have been created via `new`, or usage of it has already
    /// initialized the internal handles.
    ///
    /// This is a dangerous function to call as both this mutex and the returned
    /// mutex will share the same handles to the underlying mutex/condition
    /// variable. Care must be taken to ensure that deallocation happens
    /// accordingly.
    pub unsafe fn clone(&self) -> Mutex {
        let lock = self.lock.load(atomics::Relaxed);
        let cond = self.cond.load(atomics::Relaxed);
        assert!(lock != 0);
        assert!(cond != 0);
        Mutex {
            lock: atomics::AtomicUint::new(lock),
            cond: atomics::AtomicUint::new(cond),
        }
    }

    /// Acquires this lock. This assumes that the current thread does not
    /// already hold the lock.
    pub unsafe fn lock(&mut self) { imp::lock(self.getlock()) }

    /// Attempts to acquire the lock. The value returned is whether the lock was
    /// acquired or not
    pub unsafe fn trylock(&mut self) -> bool { imp::trylock(self.getlock()) }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    pub unsafe fn unlock(&mut self) { imp::unlock(self.getlock()) }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held
    pub unsafe fn wait(&mut self) { imp::wait(self.getcond(), self.getlock()) }

    /// Signals a thread in `wait` to wake up
    pub unsafe fn signal(&mut self) { imp::signal(self.getcond()) }

    /// This function is especially unsafe because there are no guarantees made
    /// that no other thread is currently holding the lock or waiting on the
    /// condition variable contained inside.
    pub unsafe fn destroy(&mut self) {
        let lock = self.lock.swap(0, atomics::Relaxed);
        let cond = self.cond.swap(0, atomics::Relaxed);
        if lock != 0 { imp::free_lock(lock) }
        if cond != 0 { imp::free_cond(cond) }
    }

    unsafe fn getlock(&mut self) -> *c_void {
        match self.lock.load(atomics::Relaxed) {
            0 => {}
            n => return n as *c_void
        }
        let lock = imp::init_lock();
        match self.lock.compare_and_swap(0, lock, atomics::SeqCst) {
            0 => return lock as *c_void,
            _ => {}
        }
        imp::free_lock(lock);
        return self.lock.load(atomics::Relaxed) as *c_void;
    }

    unsafe fn getcond(&mut self) -> *c_void {
        match self.cond.load(atomics::Relaxed) {
            0 => {}
            n => return n as *c_void
        }
        let cond = imp::init_cond();
        match self.cond.compare_and_swap(0, cond, atomics::SeqCst) {
            0 => return cond as *c_void,
            _ => {}
        }
        imp::free_cond(cond);
        return self.cond.load(atomics::Relaxed) as *c_void;
    }
}

#[cfg(unix)]
mod imp {
    use libc::c_void;
    use libc;
    use ptr;

    type pthread_mutex_t = libc::c_void;
    type pthread_mutexattr_t = libc::c_void;
    type pthread_cond_t = libc::c_void;
    type pthread_condattr_t = libc::c_void;

    pub unsafe fn init_lock() -> uint {
        let block = libc::malloc(rust_pthread_mutex_t_size() as libc::size_t);
        assert!(!block.is_null());
        let n = pthread_mutex_init(block, ptr::null());
        assert_eq!(n, 0);
        return block as uint;
    }

    pub unsafe fn init_cond() -> uint {
        let block = libc::malloc(rust_pthread_cond_t_size() as libc::size_t);
        assert!(!block.is_null());
        let n = pthread_cond_init(block, ptr::null());
        assert_eq!(n, 0);
        return block as uint;
    }

    pub unsafe fn free_lock(h: uint) {
        let block = h as *c_void;
        assert_eq!(pthread_mutex_destroy(block), 0);
        libc::free(block);
    }

    pub unsafe fn free_cond(h: uint) {
        let block = h as *c_void;
        assert_eq!(pthread_cond_destroy(block), 0);
        libc::free(block);
    }

    pub unsafe fn lock(l: *pthread_mutex_t) {
        assert_eq!(pthread_mutex_lock(l), 0);
    }

    pub unsafe fn trylock(l: *c_void) -> bool {
        pthread_mutex_trylock(l) == 0
    }

    pub unsafe fn unlock(l: *pthread_mutex_t) {
        assert_eq!(pthread_mutex_unlock(l), 0);
    }

    pub unsafe fn wait(cond: *pthread_cond_t, m: *pthread_mutex_t) {
        assert_eq!(pthread_cond_wait(cond, m), 0);
    }

    pub unsafe fn signal(cond: *pthread_cond_t) {
        assert_eq!(pthread_cond_signal(cond), 0);
    }

    extern {
        fn rust_pthread_mutex_t_size() -> libc::c_int;
        fn rust_pthread_cond_t_size() -> libc::c_int;
    }

    extern {
        fn pthread_mutex_init(lock: *pthread_mutex_t,
                              attr: *pthread_mutexattr_t) -> libc::c_int;
        fn pthread_mutex_destroy(lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_init(cond: *pthread_cond_t,
                              attr: *pthread_condattr_t) -> libc::c_int;
        fn pthread_cond_destroy(cond: *pthread_cond_t) -> libc::c_int;
        fn pthread_mutex_lock(lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_trylock(lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_unlock(lock: *pthread_mutex_t) -> libc::c_int;

        fn pthread_cond_wait(cond: *pthread_cond_t,
                             lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_signal(cond: *pthread_cond_t) -> libc::c_int;
    }
}

#[cfg(windows)]
mod imp {
    use libc;
    use libc::{HANDLE, BOOL, LPSECURITY_ATTRIBUTES, c_void, DWORD, LPCSTR};
    use ptr;
    type LPCRITICAL_SECTION = *c_void;
    static SPIN_COUNT: DWORD = 4000;

    pub unsafe fn init_lock() -> uint {
        let block = libc::malloc(rust_crit_section_size() as libc::size_t);
        assert!(!block.is_null());
        InitializeCriticalSectionAndSpinCount(block, SPIN_COUNT);
        return block as uint;
    }

    pub unsafe fn init_cond() -> uint {
        return CreateEventA(ptr::mut_null(), libc::FALSE, libc::FALSE,
                            ptr::null()) as uint;
    }

    pub unsafe fn free_lock(h: uint) {
        DeleteCriticalSection(h as LPCRITICAL_SECTION);
        libc::free(h as *c_void);
    }

    pub unsafe fn free_cond(h: uint) {
        let block = h as HANDLE;
        libc::CloseHandle(block);
    }

    pub unsafe fn lock(l: *c_void) {
        EnterCriticalSection(l as LPCRITICAL_SECTION)
    }

    pub unsafe fn trylock(l: *c_void) -> bool {
        TryEnterCriticalSection(l as LPCRITICAL_SECTION) != 0
    }

    pub unsafe fn unlock(l: *c_void) {
        LeaveCriticalSection(l as LPCRITICAL_SECTION)
    }

    pub unsafe fn wait(cond: *c_void, m: *c_void) {
        unlock(m);
        WaitForSingleObject(cond as HANDLE, 0);
        lock(m);
    }

    pub unsafe fn signal(cond: *c_void) {
        assert!(SetEvent(cond as HANDLE) != 0);
    }

    extern {
        fn rust_crit_section_size() -> libc::c_int;
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
    use super::{Mutex, MUTEX_INIT};
    use rt::thread::Thread;

    #[test]
    fn somke_lock() {
        static mut lock: Mutex = MUTEX_INIT;
        unsafe {
            lock.lock();
            lock.unlock();
        }
    }

    #[test]
    fn somke_cond() {
        static mut lock: Mutex = MUTEX_INIT;
        unsafe {
            lock.lock();
            let t = do Thread::start {
                lock.lock();
                lock.signal();
                lock.unlock();
            };
            lock.wait();
            lock.unlock();
            t.join();
        }
    }

    #[test]
    fn destroy_immediately() {
        unsafe {
            let mut m = Mutex::empty();
            m.destroy();
        }
    }
}
