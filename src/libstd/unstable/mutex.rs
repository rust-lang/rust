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
    pub unsafe fn lock(&self) { self.inner.lock() }

    /// Attempts to acquire the lock. The value returned is whether the lock was
    /// acquired or not
    pub unsafe fn trylock(&self) -> bool { self.inner.trylock() }

    /// Unlocks the lock. This assumes that the current thread already holds the
    /// lock.
    pub unsafe fn unlock(&self) { self.inner.unlock() }

    /// This function is especially unsafe because there are no guarantees made
    /// that no other thread is currently holding the lock or waiting on the
    /// condition variable contained inside.
    pub unsafe fn destroy(&self) { self.inner.destroy() }
}

pub struct Cond {
    priv inner: imp::Cond,
}

pub static COND_INIT: Cond = Cond {
    inner: imp::COND_INIT,
};

impl Cond {
    /// Creates a new condition variable
    pub unsafe fn new() -> Cond {
        Cond { inner: imp::Cond::new() }
    }

    /// Block on the internal condition variable.
    ///
    /// This function assumes that the lock is already held
    pub unsafe fn wait(&self, mutex: &Mutex) { self.inner.wait(&mutex.inner) }

    /// Signals a thread in `wait` to wake up
    pub unsafe fn signal(&self) { self.inner.signal() }

    /// This function is especially unsafe because there are no guarantees made
    /// that no other thread is currently holding the lock or waiting on the
    /// condition variable contained inside.
    pub unsafe fn destroy(&self) { self.inner.destroy() }
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
    #[cfg(target_os = "android")]
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

    #[no_freeze]
    pub struct Mutex {
        priv lock: pthread_mutex_t,
    }

    pub static MUTEX_INIT: Mutex = Mutex {
        lock: PTHREAD_MUTEX_INITIALIZER,
    };

    impl Mutex {
        pub unsafe fn new() -> Mutex {
            let m = Mutex {
                lock: intrinsics::init(),
            };

            pthread_mutex_init(&m.lock, 0 as *libc::c_void);

            return m;
        }

        pub unsafe fn lock(&self) { pthread_mutex_lock(&self.lock); }
        pub unsafe fn unlock(&self) { pthread_mutex_unlock(&self.lock); }
        pub unsafe fn trylock(&self) -> bool {
            pthread_mutex_trylock(&self.lock) == 0
        }
        pub unsafe fn destroy(&self) {
            pthread_mutex_destroy(&self.lock);
        }
    }

    pub struct Cond {
        priv cond: pthread_cond_t,
    }

    pub static COND_INIT: Cond = Cond {
        cond: PTHREAD_COND_INITIALIZER,
    };

    impl Cond {
        pub unsafe fn new() -> Cond {
            let c = Cond {
                cond: intrinsics::init(),
            };

            pthread_cond_init(&c.cond, 0 as *libc::c_void);

            return c;
        }

        pub unsafe fn signal(&self) { pthread_cond_signal(&self.cond); }

        pub unsafe fn wait(&self, mutex: &Mutex) {
            pthread_cond_wait(&self.cond, &mutex.lock);
        }

        pub unsafe fn destroy(&self) {
            pthread_cond_destroy(&self.cond);
        }
    }

    extern {
        fn pthread_mutex_init(lock: *pthread_mutex_t,
                              attr: *libc::c_void) -> libc::c_int;
        fn pthread_mutex_lock(lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_trylock(lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_mutex_unlock(lock: *pthread_mutex_t) -> libc::c_int;

        fn pthread_cond_init(cond: *pthread_cond_t,
                             attr: *libc::c_void) -> libc::c_int;
        fn pthread_cond_wait(cond: *pthread_cond_t,
                             lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_signal(cond: *pthread_cond_t) -> libc::c_int;
        fn pthread_mutex_destroy(lock: *pthread_mutex_t) -> libc::c_int;
        fn pthread_cond_destroy(lock: *pthread_cond_t) -> libc::c_int;
    }
}

#[cfg(windows)]
mod imp {
    use libc::{HANDLE, BOOL, LPSECURITY_ATTRIBUTES, c_void, DWORD, LPCSTR};
    use libc;
    use ptr::RawPtr;
    use ptr;
    use sync::atomics;

    type LPCRITICAL_SECTION = *c_void;
    static SPIN_COUNT: DWORD = 4000;
    #[cfg(target_arch = "x86")]
    static CRIT_SECTION_SIZE: uint = 24;

    #[no_freeze]
    pub struct Mutex {
        // pointers for the lock/cond handles, atomically updated
        priv lock: atomics::AtomicUint,
    }

    pub static MUTEX_INIT: Mutex = Mutex {
        lock: atomics::INIT_ATOMIC_UINT,
    };

    impl Mutex {
        pub unsafe fn new() -> Mutex {
            Mutex {
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

        /// This function is especially unsafe because there are no guarantees made
        /// that no other thread is currently holding the lock or waiting on the
        /// condition variable contained inside.
        pub unsafe fn destroy(&self) {
            let lock = self.lock.swap(0, atomics::Relaxed);
            if lock != 0 { free_lock(lock) }
        }

        unsafe fn getlock(&self) -> *c_void {
            match self.lock.load(atomics::Relaxed) {
                0 => {}
                n => return n as *c_void
            }
            let lock = init_lock();
            match self.lock.compare_and_swap(0, lock, atomics::SeqCst) {
                0 => return lock as *c_void,
                _ => {}
            }
            free_lock(lock);
            return self.lock.load(atomics::Relaxed) as *c_void;
        }
    }

    pub struct Cond {
        priv cond: atomics::AtomicUint,
    }

    pub static COND_INIT: Cond = Cond {
        cond: atomics::INIT_ATOMIC_UINT,
    };

    impl Cond {
        pub unsafe fn new() -> Cond {
            Cond {
                cond: atomics::AtomicUint::new(init_cond()),
            }
        }

        pub unsafe fn wait(&self, mutex: &Mutex) {
            mutex.unlock();
            WaitForSingleObject(self.getcond() as HANDLE, libc::INFINITE);
            mutex.lock();
        }

        pub unsafe fn signal(&self) {
            assert!(SetEvent(self.getcond() as HANDLE) != 0);
        }

        /// This function is especially unsafe because there are no guarantees made
        /// that no other thread is currently holding the lock or waiting on the
        /// condition variable contained inside.
        pub unsafe fn destroy(&self) {
            let cond = self.cond.swap(0, atomics::Relaxed);
            if cond != 0 { free_cond(cond) }
        }

        unsafe fn getcond(&self) -> *c_void {
            match self.cond.load(atomics::Relaxed) {
                0 => {}
                n => return n as *c_void
            }
            let cond = init_cond();
            match self.cond.compare_and_swap(0, cond, atomics::SeqCst) {
                0 => return cond as *c_void,
                _ => {}
            }
            free_cond(cond);
            return self.cond.load(atomics::Relaxed) as *c_void;
        }
    }

    pub unsafe fn init_lock() -> uint {
        let block = libc::malloc(CRIT_SECTION_SIZE as libc::size_t);
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
            // Make sure we never overflow, we'll never have int::min_value
            // simultaneous calls to `doit` to make this value go back to 0
            self.cnt.store(int::min_value, atomics::SeqCst);
            return
        }

        // If the count is negative, then someone else finished the job,
        // otherwise we run the job and record how many people will try to grab
        // this lock
        unsafe { self.mutex.lock() }
        if self.cnt.load(atomics::SeqCst) > 0 {
            f();
            let prev = self.cnt.swap(int::min_value, atomics::SeqCst);
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
    use super::{ONCE_INIT, Once, Mutex, MUTEX_INIT, Cond, COND_INIT};
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
        static lock: Mutex = MUTEX_INIT;
        unsafe {
            lock.lock();
            lock.unlock();
        }
    }

    #[test]
    fn somke_cond() {
        static lock: Mutex = MUTEX_INIT;
        static cond: Cond = COND_INIT;
        unsafe {
            lock.lock();
            let t = do Thread::start {
                lock.lock();
                cond.signal();
                lock.unlock();
            };
            cond.wait(&lock);
            lock.unlock();
            t.join();
        }
    }

    #[test]
    fn destroy_immediately() {
        unsafe {
            let m = Mutex::new();
            m.destroy();
        }
    }
}
