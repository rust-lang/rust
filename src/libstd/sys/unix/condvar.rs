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
use libc;
use sys::mutex::{self, Mutex};
use sys::sync as ffi;
use time::Duration;

pub struct Condvar { inner: UnsafeCell<ffi::pthread_cond_t> }

pub const CONDVAR_INIT: Condvar = Condvar {
    inner: UnsafeCell { value: ffi::PTHREAD_COND_INITIALIZER },
};

impl Condvar {
    #[inline]
    pub unsafe fn new() -> Condvar {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        Condvar { inner: UnsafeCell::new(ffi::PTHREAD_COND_INITIALIZER) }
    }

    #[inline]
    pub unsafe fn notify_one(&self) {
        let r = ffi::pthread_cond_signal(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        let r = ffi::pthread_cond_broadcast(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        let r = ffi::pthread_cond_wait(self.inner.get(), mutex::raw(mutex));
        debug_assert_eq!(r, 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        assert!(dur >= Duration::nanoseconds(0));

        // First, figure out what time it currently is
        let mut tv = libc::timeval { tv_sec: 0, tv_usec: 0 };
        let r = ffi::gettimeofday(&mut tv, 0 as *mut _);
        debug_assert_eq!(r, 0);

        // Offset that time with the specified duration
        let abs = Duration::seconds(tv.tv_sec as i64) +
                  Duration::microseconds(tv.tv_usec as i64) +
                  dur;
        let ns = abs.num_nanoseconds().unwrap() as u64;
        let timeout = libc::timespec {
            tv_sec: (ns / 1000000000) as libc::time_t,
            tv_nsec: (ns % 1000000000) as libc::c_long,
        };

        // And wait!
        let r = ffi::pthread_cond_timedwait(self.inner.get(), mutex::raw(mutex),
                                            &timeout);
        if r != 0 {
            debug_assert_eq!(r as int, libc::ETIMEDOUT as int);
            false
        } else {
            true
        }
    }

    #[inline]
    #[cfg(not(target_os = "dragonfly"))]
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_cond_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    #[cfg(target_os = "dragonfly")]
    pub unsafe fn destroy(&self) {
        let r = ffi::pthread_cond_destroy(self.inner.get());
        // On DragonFly pthread_cond_destroy() returns EINVAL if called on
        // a condvar that was just initialized with
        // ffi::PTHREAD_COND_INITIALIZER. Once it is used or
        // pthread_cond_init() is called, this behaviour no longer occurs.
        debug_assert!(r == 0 || r == libc::EINVAL);
    }
}
