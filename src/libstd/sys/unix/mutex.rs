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
use sys_common::mutex;

pub struct Mutex { inner: UnsafeCell<ffi::pthread_mutex_t> }

#[inline]
pub unsafe fn raw(m: &Mutex) -> *mut ffi::pthread_mutex_t {
    m.inner.get()
}

pub const MUTEX_INIT: Mutex = Mutex {
    inner: UnsafeCell { value: ffi::PTHREAD_MUTEX_INITIALIZER },
};

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
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_mutex_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }
}
