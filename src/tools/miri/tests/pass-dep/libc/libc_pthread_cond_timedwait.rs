//@ignore-target: windows # No pthreads on Windows
//@ignore-target: apple # pthread_condattr_setclock is not supported on MacOS.
//@compile-flags: -Zmiri-disable-isolation

/// Test that conditional variable timeouts are working properly with both
/// monotonic and system clocks.
use std::mem::MaybeUninit;
use std::time::Instant;

fn test_timed_wait_timeout(clock_id: i32) {
    unsafe {
        let mut attr: MaybeUninit<libc::pthread_condattr_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_condattr_init(attr.as_mut_ptr()), 0);
        assert_eq!(libc::pthread_condattr_setclock(attr.as_mut_ptr(), clock_id), 0);

        let mut cond: MaybeUninit<libc::pthread_cond_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_cond_init(cond.as_mut_ptr(), attr.as_ptr()), 0);
        assert_eq!(libc::pthread_condattr_destroy(attr.as_mut_ptr()), 0);

        let mut mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;

        let mut now_mu: MaybeUninit<libc::timespec> = MaybeUninit::uninit();
        assert_eq!(libc::clock_gettime(clock_id, now_mu.as_mut_ptr()), 0);
        let now = now_mu.assume_init();
        // Waiting for a second... mostly because waiting less requires much more tricky arithmetic.
        // FIXME: wait less.
        let timeout = libc::timespec { tv_sec: now.tv_sec + 1, tv_nsec: now.tv_nsec };

        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        let current_time = Instant::now();
        assert_eq!(
            libc::pthread_cond_timedwait(cond.as_mut_ptr(), &mut mutex as *mut _, &timeout),
            libc::ETIMEDOUT
        );
        let elapsed_time = current_time.elapsed().as_millis();
        assert!(900 <= elapsed_time && elapsed_time <= 1300);

        // Test calling `pthread_cond_timedwait` again with an already elapsed timeout.
        assert_eq!(
            libc::pthread_cond_timedwait(cond.as_mut_ptr(), &mut mutex as *mut _, &timeout),
            libc::ETIMEDOUT
        );

        // Test that invalid nanosecond values (above 10^9 or negative) are rejected with the
        // correct error code.
        let invalid_timeout_1 = libc::timespec { tv_sec: now.tv_sec + 1, tv_nsec: 1_000_000_000 };
        assert_eq!(
            libc::pthread_cond_timedwait(
                cond.as_mut_ptr(),
                &mut mutex as *mut _,
                &invalid_timeout_1
            ),
            libc::EINVAL
        );
        let invalid_timeout_2 = libc::timespec { tv_sec: now.tv_sec + 1, tv_nsec: -1 };
        assert_eq!(
            libc::pthread_cond_timedwait(
                cond.as_mut_ptr(),
                &mut mutex as *mut _,
                &invalid_timeout_2
            ),
            libc::EINVAL
        );
        // Test that invalid second values (negative) are rejected with the correct error code.
        let invalid_timeout_3 = libc::timespec { tv_sec: -1, tv_nsec: 0 };
        assert_eq!(
            libc::pthread_cond_timedwait(
                cond.as_mut_ptr(),
                &mut mutex as *mut _,
                &invalid_timeout_3
            ),
            libc::EINVAL
        );

        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_cond_destroy(cond.as_mut_ptr()), 0);
    }
}

fn main() {
    test_timed_wait_timeout(libc::CLOCK_MONOTONIC);
    test_timed_wait_timeout(libc::CLOCK_REALTIME);
}
