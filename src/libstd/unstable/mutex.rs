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

pub struct Mutex {
    priv inner: imp::Mutex,
}

pub static MUTEX_INIT: Mutex = Mutex {
    inner: imp::MUTEX_INIT,
};

impl Mutex {
    /// Creates a new mutex
    pub unsafe fn new() -> Mutex {
        Mutex { inner: imp::Mutex::new() }
    }

    /// Acquires this lock. This assumes that the current thread does not
    /// already hold the lock.
    pub unsafe fn lock(&mut self) { self.inner.lock() }

    /// Attempts to acquire the lock. The value returned is whether the lock was
    /// acquired or not
    pub unsafe fn trylock(&mut self) -> bool { self.inner.trylock() }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    pub unsafe fn unlock(&mut self) { self.inner.unlock() }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held
    pub unsafe fn wait(&mut self) { self.inner.wait() }

    /// Signals a thread in `wait` to wake up
    pub unsafe fn signal(&mut self) { self.inner.signal() }

    /// This function is especially unsafe because there are no guarantees made
    /// that no other thread is currently holding the lock or waiting on the
    /// condition variable contained inside.
    pub unsafe fn destroy(&mut self) { self.inner.destroy() }
}

#[cfg(unix)]
mod imp {
    use libc;
    use self::os::{PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER,
                   pthread_mutex_t, pthread_cond_t};
    use unstable::intrinsics;

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
        #[cfg(target_arch = "x86_64")]
        static __SIZEOF_PTHREAD_COND_T: uint = 48 - 8;
        #[cfg(target_arch = "x86")]
        static __SIZEOF_PTHREAD_COND_T: uint = 48 - 8;

        pub struct pthread_mutex_t {
            __align: libc::c_long,
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
        priv lock: pthread_mutex_t,
        priv cond: pthread_cond_t,
    }

    pub static MUTEX_INIT: Mutex = Mutex {
        lock: PTHREAD_MUTEX_INITIALIZER,
        cond: PTHREAD_COND_INITIALIZER,
    };

    impl Mutex {
        pub unsafe fn new() -> Mutex {
            let mut m = Mutex {
                lock: intrinsics::init(),
                cond: intrinsics::init(),
            };

            pthread_mutex_init(&mut m.lock, 0 as *libc::c_void);
            pthread_cond_init(&mut m.cond, 0 as *libc::c_void);

            return m;
        }

        pub unsafe fn lock(&mut self) { pthread_mutex_lock(&mut self.lock); }
        pub unsafe fn unlock(&mut self) { pthread_mutex_unlock(&mut self.lock); }
        pub unsafe fn signal(&mut self) { pthread_cond_signal(&mut self.cond); }
        pub unsafe fn wait(&mut self) {
            pthread_cond_wait(&mut self.cond, &mut self.lock);
        }
        pub unsafe fn trylock(&mut self) -> bool {
            pthread_mutex_trylock(&mut self.lock) == 0
        }
        pub unsafe fn destroy(&mut self) {
            pthread_mutex_destroy(&mut self.lock);
            pthread_cond_destroy(&mut self.cond);
        }
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
    use rt::global_heap::malloc_raw;
    use libc::{HANDLE, BOOL, LPSECURITY_ATTRIBUTES, c_void, DWORD, LPCSTR};
    use libc;
    use ptr;
    use sync::atomics;

    type LPCRITICAL_SECTION = *mut c_void;
    static SPIN_COUNT: DWORD = 4000;
    #[cfg(target_arch = "x86")]
    static CRIT_SECTION_SIZE: uint = 24;

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
        pub unsafe fn new() -> Mutex {
            Mutex {
                lock: atomics::AtomicUint::new(init_lock()),
                cond: atomics::AtomicUint::new(init_cond()),
            }
        }
        pub unsafe fn lock(&mut self) {
            EnterCriticalSection(self.getlock() as LPCRITICAL_SECTION)
        }
        pub unsafe fn trylock(&mut self) -> bool {
            TryEnterCriticalSection(self.getlock() as LPCRITICAL_SECTION) != 0
        }
        pub unsafe fn unlock(&mut self) {
            LeaveCriticalSection(self.getlock() as LPCRITICAL_SECTION)
        }

        pub unsafe fn wait(&mut self) {
            self.unlock();
            WaitForSingleObject(self.getcond() as HANDLE, libc::INFINITE);
            self.lock();
        }

        pub unsafe fn signal(&mut self) {
            assert!(SetEvent(self.getcond() as HANDLE) != 0);
        }

        /// This function is especially unsafe because there are no guarantees made
        /// that no other thread is currently holding the lock or waiting on the
        /// condition variable contained inside.
        pub unsafe fn destroy(&mut self) {
            let lock = self.lock.swap(0, atomics::SeqCst);
            let cond = self.cond.swap(0, atomics::SeqCst);
            if lock != 0 { free_lock(lock) }
            if cond != 0 { free_cond(cond) }
        }

        unsafe fn getlock(&mut self) -> *mut c_void {
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

        unsafe fn getcond(&mut self) -> *mut c_void {
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

    use super::{Mutex, MUTEX_INIT};
    use rt::thread::Thread;
    use task;

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
            let t = Thread::start(proc() {
                lock.lock();
                lock.signal();
                lock.unlock();
            });
            lock.wait();
            lock.unlock();
            t.join();
        }
    }

    #[test]
    fn destroy_immediately() {
        unsafe {
            let mut m = Mutex::new();
            m.destroy();
        }
    }
}
