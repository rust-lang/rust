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
use time::Duration;

pub struct Condvar { inner: UnsafeCell<libc::pthread_cond_t> }

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

const TIMESPEC_MAX: libc::timespec = libc::timespec {
    tv_sec: <libc::time_t>::max_value(),
    tv_nsec: 1_000_000_000 - 1,
};

impl Condvar {
    pub const fn new() -> Condvar {
        // Might be moved and address is changing it is better to avoid
        // initialization of potentially opaque OS data before it landed
        Condvar { inner: UnsafeCell::new(libc::PTHREAD_COND_INITIALIZER) }
    }

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
    pub unsafe fn init(&mut self) {}

    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
    pub unsafe fn init(&mut self) {
        use mem;
        let mut attr: libc::pthread_condattr_t = mem::uninitialized();
        let r = libc::pthread_condattr_init(&mut attr);
        assert_eq!(r, 0);
        let r = libc::pthread_condattr_setclock(&mut attr, libc::CLOCK_MONOTONIC);
        assert_eq!(r, 0);
        let r = libc::pthread_cond_init(self.inner.get(), &attr);
        assert_eq!(r, 0);
        let r = libc::pthread_condattr_destroy(&mut attr);
        assert_eq!(r, 0);
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

    // This implementation is used on systems that support pthread_condattr_setclock
    // where we configure condition variable to use monotonic clock (instead of
    // default system clock). This approach avoids all problems that result
    // from changes made to the system time.
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "android")))]
    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        use mem;

        let mut now: libc::timespec = mem::zeroed();
        let r = libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now);
        assert_eq!(r, 0);

        // Nanosecond calculations can't overflow because both values are below 1e9.
        let nsec = dur.subsec_nanos() as libc::c_long + now.tv_nsec as libc::c_long;
        // FIXME: Casting u64 into time_t could truncate the value.
        let sec = (dur.as_secs() as libc::time_t)
            .checked_add((nsec / 1_000_000_000) as libc::time_t)
            .and_then(|s| s.checked_add(now.tv_sec));
        let nsec = nsec % 1_000_000_000;

        let timeout = sec.map(|s| {
            libc::timespec { tv_sec: s, tv_nsec: nsec }
        }).unwrap_or(TIMESPEC_MAX);

        let r = libc::pthread_cond_timedwait(self.inner.get(), mutex::raw(mutex),
                                            &timeout);
        assert!(r == libc::ETIMEDOUT || r == 0);
        r == 0
    }


    // This implementation is modeled after libcxx's condition_variable
    // https://github.com/llvm-mirror/libcxx/blob/release_35/src/condition_variable.cpp#L46
    // https://github.com/llvm-mirror/libcxx/blob/release_35/include/__mutex_base#L367
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "android"))]
    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        use ptr;
        use time::Instant;

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
        }).unwrap_or(TIMESPEC_MAX);

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
