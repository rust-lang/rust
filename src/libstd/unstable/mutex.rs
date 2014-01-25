// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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

use int;
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
            lock: atomics::AtomicUint::new(imp::init_lock()),
            cond: atomics::AtomicUint::new(imp::init_cond()),
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

    unsafe fn getlock(&mut self) -> uint{
        match self.lock.load(atomics::Relaxed) {
            0 => {}
            n => return n
        }
        let lock = imp::init_lock();
        match self.lock.compare_and_swap(0, lock, atomics::SeqCst) {
            0 => return lock,
            _ => {}
        }
        imp::free_lock(lock);
        self.lock.load(atomics::Relaxed)
    }

    unsafe fn getcond(&mut self) -> uint {
        match self.cond.load(atomics::Relaxed) {
            0 => {}
            n => return n
        }
        let cond = imp::init_cond();
        match self.cond.compare_and_swap(0, cond, atomics::SeqCst) {
            0 => return cond,
            _ => {}
        }
        imp::free_cond(cond);
        self.cond.load(atomics::Relaxed)
    }
}

#[cfg(unix)]
mod imp {
    use libc;
    use ptr;
    use rt::global_heap::malloc_raw;

    type pthread_mutex_t = libc::c_void;
    type pthread_mutexattr_t = libc::c_void;
    type pthread_cond_t = libc::c_void;
    type pthread_condattr_t = libc::c_void;

    pub unsafe fn init_lock() -> uint {
        let block = malloc_raw(rust_pthread_mutex_t_size() as uint) as *mut pthread_mutex_t;
        let n = pthread_mutex_init(block, ptr::null());
        assert_eq!(n, 0);
        return block as uint;
    }

    pub unsafe fn init_cond() -> uint {
        let block = malloc_raw(rust_pthread_cond_t_size() as uint) as *mut pthread_cond_t;
        let n = pthread_cond_init(block, ptr::null());
        assert_eq!(n, 0);
        return block as uint;
    }

    pub unsafe fn free_lock(h: uint) {
        let block = h as *mut libc::c_void;
        assert_eq!(pthread_mutex_destroy(block), 0);
        libc::free(block);
    }

    pub unsafe fn free_cond(h: uint) {
        let block = h as *mut pthread_cond_t;
        assert_eq!(pthread_cond_destroy(block), 0);
        libc::free(block);
    }

    pub unsafe fn lock(l: uint) {
        assert_eq!(pthread_mutex_lock(l as *mut pthread_mutex_t), 0);
    }

    pub unsafe fn trylock(l: uint) -> bool {
        pthread_mutex_trylock(l as *mut pthread_mutex_t) == 0
    }

    pub unsafe fn unlock(l: uint) {
        assert_eq!(pthread_mutex_unlock(l as *mut pthread_mutex_t), 0);
    }

    pub unsafe fn wait(cond: uint, m: uint) {
        assert_eq!(pthread_cond_wait(cond as *mut pthread_cond_t, m as *mut pthread_mutex_t), 0);
    }

    pub unsafe fn signal(cond: uint) {
        assert_eq!(pthread_cond_signal(cond as *mut pthread_cond_t), 0);
    }

    extern {
        fn rust_pthread_mutex_t_size() -> libc::c_int;
        fn rust_pthread_cond_t_size() -> libc::c_int;
    }

    extern {
        fn pthread_mutex_init(lock: *mut pthread_mutex_t,
                              attr: *pthread_mutexattr_t) -> libc::c_int;
        fn pthread_mutex_destroy(lock: *mut pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_init(cond: *mut pthread_cond_t,
                              attr: *pthread_condattr_t) -> libc::c_int;
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
    use libc;
    use libc::{HANDLE, BOOL, LPSECURITY_ATTRIBUTES, c_void, DWORD, LPCSTR};
    use ptr;
    use rt::global_heap::malloc_raw;

    type LPCRITICAL_SECTION = *c_void;
    static SPIN_COUNT: DWORD = 4000;

    pub unsafe fn init_lock() -> uint {
        let block = malloc_raw(rust_crit_section_size() as uint) as *c_void;
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

    pub unsafe fn lock(l: uint) {
        EnterCriticalSection(l as LPCRITICAL_SECTION)
    }

    pub unsafe fn trylock(l: uint) -> bool {
        TryEnterCriticalSection(l as LPCRITICAL_SECTION) != 0
    }

    pub unsafe fn unlock(l: uint) {
        LeaveCriticalSection(l as LPCRITICAL_SECTION)
    }

    pub unsafe fn wait(cond: uint, m: uint) {
        unlock(m);
        WaitForSingleObject(cond as HANDLE, libc::INFINITE);
        lock(m);
    }

    pub unsafe fn signal(cond: uint) {
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

/// A type which can be used to run a one-time global initialization. This type
/// is *unsafe* to use because it is built on top of the `Mutex` in this module.
/// It does not know whether the currently running task is in a green or native
/// context, and a blocking mutex should *not* be used under normal
/// circumstances on a green task.
///
/// Despite its unsafety, it is often useful to have a one-time initialization
/// routine run for FFI bindings or related external functionality. This type
/// can only be statically constructed with the `ONCE_INIT` value.
///
/// # Example
///
/// ```rust
/// use std::unstable::mutex::{Once, ONCE_INIT};
///
/// static mut START: Once = ONCE_INIT;
/// unsafe {
///     START.doit(|| {
///         // run initialization here
///     });
/// }
/// ```
pub struct Once {
    priv mutex: Mutex,
    priv cnt: atomics::AtomicInt,
    priv lock_cnt: atomics::AtomicInt,
}

/// Initialization value for static `Once` values.
pub static ONCE_INIT: Once = Once {
    mutex: MUTEX_INIT,
    cnt: atomics::INIT_ATOMIC_INT,
    lock_cnt: atomics::INIT_ATOMIC_INT,
};

impl Once {
    /// Perform an initialization routine once and only once. The given closure
    /// will be executed if this is the first time `doit` has been called, and
    /// otherwise the routine will *not* be invoked.
    ///
    /// This method will block the calling *os thread* if another initialization
    /// routine is currently running.
    ///
    /// When this function returns, it is guaranteed that some initialization
    /// has run and completed (it may not be the closure specified).
    pub fn doit(&mut self, f: ||) {
        // Implementation-wise, this would seem like a fairly trivial primitive.
        // The stickler part is where our mutexes currently require an
        // allocation, and usage of a `Once` should't leak this allocation.
        //
        // This means that there must be a deterministic destroyer of the mutex
        // contained within (because it's not needed after the initialization
        // has run).
        //
        // The general scheme here is to gate all future threads once
        // initialization has completed with a "very negative" count, and to
        // allow through threads to lock the mutex if they see a non negative
        // count. For all threads grabbing the mutex, exactly one of them should
        // be responsible for unlocking the mutex, and this should only be done
        // once everyone else is done with the mutex.
        //
        // This atomicity is achieved by swapping a very negative value into the
        // shared count when the initialization routine has completed. This will
        // read the number of threads which will at some point attempt to
        // acquire the mutex. This count is then squirreled away in a separate
        // variable, and the last person on the way out of the mutex is then
        // responsible for destroying the mutex.
        //
        // It is crucial that the negative value is swapped in *after* the
        // initialization routine has completed because otherwise new threads
        // calling `doit` will return immediately before the initialization has
        // completed.

        let prev = self.cnt.fetch_add(1, atomics::SeqCst);
        if prev < 0 {
            // Make sure we never overflow, we'll never have int::MIN
            // simultaneous calls to `doit` to make this value go back to 0
            self.cnt.store(int::MIN, atomics::SeqCst);
            return
        }

        // If the count is negative, then someone else finished the job,
        // otherwise we run the job and record how many people will try to grab
        // this lock
        unsafe { self.mutex.lock() }
        if self.cnt.load(atomics::SeqCst) > 0 {
            f();
            let prev = self.cnt.swap(int::MIN, atomics::SeqCst);
            self.lock_cnt.store(prev, atomics::SeqCst);
        }
        unsafe { self.mutex.unlock() }

        // Last one out cleans up after everyone else, no leaks!
        if self.lock_cnt.fetch_add(-1, atomics::SeqCst) == 1 {
            unsafe { self.mutex.destroy() }
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    use rt::thread::Thread;
    use super::{ONCE_INIT, Once, Mutex, MUTEX_INIT};
    use task;

    #[test]
    fn smoke_once() {
        static mut o: Once = ONCE_INIT;
        let mut a = 0;
        unsafe { o.doit(|| a += 1); }
        assert_eq!(a, 1);
        unsafe { o.doit(|| a += 1); }
        assert_eq!(a, 1);
    }

    #[test]
    fn stampede_once() {
        static mut o: Once = ONCE_INIT;
        static mut run: bool = false;

        let (p, c) = SharedChan::new();
        for _ in range(0, 10) {
            let c = c.clone();
            do spawn {
                for _ in range(0, 4) { task::deschedule() }
                unsafe {
                    o.doit(|| {
                        assert!(!run);
                        run = true;
                    });
                    assert!(run);
                }
                c.send(());
            }
        }

        unsafe {
            o.doit(|| {
                assert!(!run);
                run = true;
            });
            assert!(run);
        }

        for _ in range(0, 10) {
            p.recv();
        }
    }

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
