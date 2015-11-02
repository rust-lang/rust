// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use cell::UnsafeCell;
use sys::unix::sync as ffi;

pub struct RwLock { inner: UnsafeCell<ffi::pthread_rwlock_t> }

unsafe impl Send for RwLock {}
unsafe impl Sync for RwLock {}

impl RwLock {
    pub const fn new() -> RwLock { RwLock { inner: UnsafeCell::new(ffi::PTHREAD_RWLOCK_INITIALIZER) } }
}

impl RwLock {
    #[inline]
    pub unsafe fn read(&self) {
        let r = ffi::pthread_rwlock_rdlock(self.inner.get());

        // According to the pthread_rwlock_rdlock spec, this function **may**
        // fail with EDEADLK if a deadlock is detected. On the other hand
        // pthread mutexes will *never* return EDEADLK if they are initialized
        // as the "fast" kind (which ours always are). As a result, a deadlock
        // situation may actually return from the call to pthread_rwlock_rdlock
        // instead of blocking forever (as mutexes and Windows rwlocks do). Note
        // that not all unix implementations, however, will return EDEADLK for
        // their rwlocks.
        //
        // We roughly maintain the deadlocking behavior by panicking to ensure
        // that this lock acquisition does not succeed.
        if r == libc::EDEADLK {
            panic!("rwlock read lock would result in deadlock");
        } else {
            debug_assert_eq!(r, 0);
        }
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        ffi::pthread_rwlock_tryrdlock(self.inner.get()) == 0
    }
    #[inline]
    pub unsafe fn write(&self) {
        let r = ffi::pthread_rwlock_wrlock(self.inner.get());
        // see comments above for why we check for EDEADLK
        if r == libc::EDEADLK {
            panic!("rwlock write lock would result in deadlock");
        } else {
            debug_assert_eq!(r, 0);
        }
    }
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        ffi::pthread_rwlock_trywrlock(self.inner.get()) == 0
    }
    #[inline]
    pub unsafe fn read_unlock(&self) {
        let r = ffi::pthread_rwlock_unlock(self.inner.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn write_unlock(&self) { self.read_unlock() }
    #[inline]
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_rwlock_destroy(self.inner.get());
        // On DragonFly pthread_rwlock_destroy() returns EINVAL if called on a
        // rwlock that was just initialized with
        // ffi::PTHREAD_RWLOCK_INITIALIZER. Once it is used (locked/unlocked)
        // or pthread_rwlock_init() is called, this behaviour no longer occurs.
        if cfg!(target_os = "dragonfly") {
            debug_assert!(r == 0 || r == libc::EINVAL);
        } else {
            debug_assert_eq!(r, 0);
        }
    }
}
