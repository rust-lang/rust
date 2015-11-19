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
use ptr;
use sys::mutex::{self, Mutex};
use time::{Instant, Duration};

pub struct Condvar { inner: UnsafeCell<libc::pthread_cond_t> }

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    pub const fn new() -> Condvar {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        Condvar { inner: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER) }
    }

    #[inline]
    pub unsafe fn notify_one(&self) {
        let r = libc::pthread_cond_signal(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        let r = libc::pthread_cond_broadcast(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        let r = libc::pthread_cond_wait(self.inner.get(), mutex::raw(mutex));
        debug_assert_eq!(r, 0);
    }

    // This implementation is modeled after libcxx's condition_variable
    // https://github.com/llvm-mirror/libcxx/blob/release_35/src/condition_variable.cpp#L46
    // https://github.com/llvm-mirror/libcxx/blob/release_35/include/__mutex_base#L367
    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        // First, figure out what time it currently is, in both system and
        // stable time.  pthread_cond_timedwait uses system time, but we want to
        // report timeout based on stable time.
        let mut sys_now = libc::timeval { tv_sec: 0, tv_usec: 0 };
        let stable_now = Instant::now();
        let r = libc::gettimeofday(&mut sys_now, ptr::null_mut());
        debug_assert_eq!(r, 0);

        let nsec = dur.subsec_nanos() as libc::c_long +
                   (sys_now.tv_usec * 1000) as libc::c_long;
        let extra = (nsec / 1_000_000_000) as libc::time_t;
        let nsec = nsec % 1_000_000_000;
        let seconds = dur.as_secs() as libc::time_t;

        let timeout = sys_now.tv_sec.checked_add(extra).and_then(|s| {
            s.checked_add(seconds)
        }).map(|s| {
            libc::timespec { tv_sec: s, tv_nsec: nsec }
        }).unwrap_or_else(|| {
            libc::timespec {
                tv_sec: <libc::time_t>::max_value(),
                tv_nsec: 1_000_000_000 - 1,
            }
        });

        // And wait!
        let r = libc::pthread_cond_timedwait(self.inner.get(), mutex::raw(mutex),
                                            &timeout);
        debug_assert!(r == libc::ETIMEDOUT || r == 0);

        // ETIMEDOUT is not a totally reliable method of determining timeout due
        // to clock shifts, so do the check ourselves
        stable_now.elapsed() < dur
    }

    #[inline]
    #[cfg(not(target_os = "dragonfly"))]
    pub unsafe fn destroy(&self) {
        let r = libc::pthread_cond_destroy(self.inner.get());
        debug_assert_eq!(r, 0);
    }

    #[inline]
    #[cfg(target_os = "dragonfly")]
    pub unsafe fn destroy(&self) {
        let r = libc::pthread_cond_destroy(self.inner.get());
        // On DragonFly pthread_cond_destroy() returns EINVAL if called on
        // a condvar that was just initialized with
        // libc::PTHREAD_COND_INITIALIZER. Once it is used or
        // pthread_cond_init() is called, this behaviour no longer occurs.
        debug_assert!(r == 0 || r == libc::EINVAL);
    }
}
