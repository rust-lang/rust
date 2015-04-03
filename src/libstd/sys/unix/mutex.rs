// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use cell::UnsafeCell;
use sys::sync as ffi;
use mem;

pub struct Mutex { inner: UnsafeCell<ffi::pthread_mutex_t> }

#[inline]
pub unsafe fn raw(m: &Mutex) -> *mut ffi::pthread_mutex_t {
    m.inner.get()
}

pub const MUTEX_INIT: Mutex = Mutex {
    inner: UnsafeCell { value: ffi::PTHREAD_MUTEX_INITIALIZER },
};

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

#[allow(dead_code)] // sys isn't exported yet
impl Mutex {
    #[inline]
    pub unsafe fn new() -> Mutex {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        MUTEX_INIT
    }
    #[inline]
    pub unsafe fn lock(&self) {
        let r = ffi::pthread_mutex_lock(self.inner.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        let r = ffi::pthread_mutex_unlock(self.inner.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::pthread_mutex_trylock(self.inner.get()) == 0
    }
    #[inline]
    #[cfg(not(target_os = "dragonfly"))]
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_mutex_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    #[cfg(target_os = "dragonfly")]
    pub unsafe fn destroy(&self) {
        use libc;
        let r = ffi::pthread_mutex_destroy(self.inner.get());
        // On DragonFly pthread_mutex_destroy() returns EINVAL if called on a
        // mutex that was just initialized with ffi::PTHREAD_MUTEX_INITIALIZER.
        // Once it is used (locked/unlocked) or pthread_mutex_init() is called,
        // this behaviour no longer occurs.
        debug_assert!(r == 0 || r == libc::EINVAL);
    }
}

// FIXME: remove the box, because box happens twice now, once at the common layer and once here.
// Box is necessary here, because mutex may not change address after it is intialised on some
// platforms. Regular Mutex above handles this by offloading intialisation to the OS on first lock.
// Sadly, as far as reentrant mutexes go, this scheme is not quite portable and we must initialise
// when we create the mutex, in the `new`.
pub struct ReentrantMutex { inner: Box<UnsafeCell<ffi::pthread_mutex_t>> }

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub unsafe fn new() -> ReentrantMutex {
        let mutex = ReentrantMutex { inner: box mem::uninitialized() };
        let mut attr: ffi::pthread_mutexattr_t = mem::uninitialized();
        let result = ffi::pthread_mutexattr_init(&mut attr as *mut _);
        debug_assert_eq!(result, 0);
        let result = ffi::pthread_mutexattr_settype(&mut attr as *mut _,
                                                    ffi::PTHREAD_MUTEX_RECURSIVE);
        debug_assert_eq!(result, 0);
        let result = ffi::pthread_mutex_init(mutex.inner.get(), &attr as *const _);
        debug_assert_eq!(result, 0);
        let result = ffi::pthread_mutexattr_destroy(&mut attr as *mut _);
        debug_assert_eq!(result, 0);
        mutex
    }

    pub unsafe fn lock(&self) {
        let result = ffi::pthread_mutex_lock(self.inner.get());
        debug_assert_eq!(result, 0);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::pthread_mutex_trylock(self.inner.get()) == 0
    }

    pub unsafe fn unlock(&self) {
        let result = ffi::pthread_mutex_unlock(self.inner.get());
        debug_assert_eq!(result, 0);
    }

    pub unsafe fn destroy(&self) {
        let result = ffi::pthread_mutex_destroy(self.inner.get());
        debug_assert_eq!(result, 0);
    }
}
