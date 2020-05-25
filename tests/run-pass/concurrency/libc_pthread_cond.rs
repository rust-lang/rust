// ignore-windows: No libc on Windows
// ignore-macos: pthread_condattr_setclock is not supported on MacOS.
// compile-flags: -Zmiri-disable-isolation

#![feature(rustc_private)]

/// Test that conditional variable timeouts are working properly with both
/// monotonic and system clocks.
extern crate libc;

use std::mem;
use std::time::Instant;

fn test_timed_wait_timeout(clock_id: i32) {
    unsafe {
        let mut attr: libc::pthread_condattr_t = mem::zeroed();
        assert_eq!(libc::pthread_condattr_init(&mut attr as *mut _), 0);
        assert_eq!(libc::pthread_condattr_setclock(&mut attr as *mut _, clock_id), 0);

        let mut cond: libc::pthread_cond_t = mem::zeroed();
        assert_eq!(libc::pthread_cond_init(&mut cond as *mut _, &attr as *const _), 0);
        assert_eq!(libc::pthread_condattr_destroy(&mut attr as *mut _), 0);

        let mut mutex: libc::pthread_mutex_t = mem::zeroed();

        let mut now: libc::timespec = mem::zeroed();
        assert_eq!(libc::clock_gettime(clock_id, &mut now), 0);
        let timeout = libc::timespec { tv_sec: now.tv_sec + 1, tv_nsec: now.tv_nsec };

        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        let current_time = Instant::now();
        assert_eq!(
            libc::pthread_cond_timedwait(&mut cond as *mut _, &mut mutex as *mut _, &timeout),
            libc::ETIMEDOUT
        );
        let elapsed_time = current_time.elapsed().as_millis();
        assert!(900 <= elapsed_time && elapsed_time <= 1300);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_cond_destroy(&mut cond as *mut _), 0);
    }
}

fn main() {
    test_timed_wait_timeout(libc::CLOCK_MONOTONIC);
    test_timed_wait_timeout(libc::CLOCK_REALTIME);
}
