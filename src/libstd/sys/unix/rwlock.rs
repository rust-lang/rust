// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::UnsafeCell;
use sys::sync as ffi;

pub struct RWLock { inner: UnsafeCell<ffi::pthread_rwlock_t> }

pub const RWLOCK_INIT: RWLock = RWLock {
    inner: UnsafeCell { value: ffi::PTHREAD_RWLOCK_INITIALIZER },
};

impl RWLock {
    #[inline]
    pub unsafe fn new() -> RWLock {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        RWLOCK_INIT
    }
    #[inline]
    pub unsafe fn read(&self) {
        let r = ffi::pthread_rwlock_rdlock(self.inner.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        ffi::pthread_rwlock_tryrdlock(self.inner.get()) == 0
    }
    #[inline]
    pub unsafe fn write(&self) {
        let r = ffi::pthread_rwlock_wrlock(self.inner.get());
        debug_assert_eq!(r, 0);
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
    #[cfg(not(target_os = "dragonfly"))]
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_rwlock_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    #[cfg(target_os = "dragonfly")]
    pub unsafe fn destroy(&self) {
        use libc;
        let r = ffi::pthread_rwlock_destroy(self.inner.get());
        // On DragonFly pthread_rwlock_destroy() returns EINVAL if called on a
        // rwlock that was just initialized with
        // ffi::PTHREAD_RWLOCK_INITIALIZER. Once it is used (locked/unlocked)
        // or pthread_rwlock_init() is called, this behaviour no longer occurs.
        debug_assert!(r == 0 || r == libc::EINVAL);
    }
}
