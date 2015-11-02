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
use sys::unix::sync as ffi;
use mem;

pub struct Mutex(UnsafeCell<ffi::pthread_mutex_t>);
impl_inner!(Mutex(UnsafeCell<ffi::pthread_mutex_t>): AsInner);

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Mutex {
    // Might be moved and address is changing it is better to avoid
    // initialization of potentially opaque OS data before it landed
    pub const fn new() -> Mutex { Mutex(UnsafeCell::new(ffi::PTHREAD_MUTEX_INITIALIZER)) }
}

impl Mutex {
    #[inline]
    pub unsafe fn lock(&self) {
        let r = ffi::pthread_mutex_lock(self.0.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        let r = ffi::pthread_mutex_unlock(self.0.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::pthread_mutex_trylock(self.0.get()) == 0
    }
    #[inline]
    #[cfg(not(target_os = "dragonfly"))]
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_mutex_destroy(self.0.get());
        debug_assert_eq!(r, 0);
    }
    #[inline]
    #[cfg(target_os = "dragonfly")]
    pub unsafe fn destroy(&self) {
        use libc;
        let r = ffi::pthread_mutex_destroy(self.0.get());
        // On DragonFly pthread_mutex_destroy() returns EINVAL if called on a
        // mutex that was just initialized with ffi::PTHREAD_MUTEX_INITIALIZER.
        // Once it is used (locked/unlocked) or pthread_mutex_init() is called,
        // this behaviour no longer occurs.
        debug_assert!(r == 0 || r == libc::EINVAL);
    }
}

pub struct ReentrantMutex(UnsafeCell<ffi::pthread_mutex_t>);

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub const fn uninitialized() -> ReentrantMutex { ReentrantMutex(UnsafeCell::new(ffi::PTHREAD_MUTEX_INITIALIZER)) }
}

impl ReentrantMutex {
    pub unsafe fn init(&mut self) {
        let mut attr: ffi::pthread_mutexattr_t = mem::uninitialized();
        let result = ffi::pthread_mutexattr_init(&mut attr as *mut _);
        debug_assert_eq!(result, 0);
        let result = ffi::pthread_mutexattr_settype(&mut attr as *mut _,
                                                    ffi::PTHREAD_MUTEX_RECURSIVE);
        debug_assert_eq!(result, 0);
        let result = ffi::pthread_mutex_init(self.0.get(), &attr as *const _);
        debug_assert_eq!(result, 0);
        let result = ffi::pthread_mutexattr_destroy(&mut attr as *mut _);
        debug_assert_eq!(result, 0);
    }
}

impl ReentrantMutex {
    pub unsafe fn lock(&self) {
        let result = ffi::pthread_mutex_lock(self.0.get());
        debug_assert_eq!(result, 0);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::pthread_mutex_trylock(self.0.get()) == 0
    }

    pub unsafe fn unlock(&self) {
        let result = ffi::pthread_mutex_unlock(self.0.get());
        debug_assert_eq!(result, 0);
    }

    pub unsafe fn destroy(&self) {
        let result = ffi::pthread_mutex_destroy(self.0.get());
        debug_assert_eq!(result, 0);
    }
}
