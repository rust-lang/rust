//@ignore-target: windows # No pthreads on Windows
//@compile-flags: -Zmiri-disable-isolation
//@run-native

/// Test that conditional variable timeouts are working properly with both
/// monotonic and system clocks.
use std::mem::MaybeUninit;
use std::time::Instant;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::errno_check;

fn test_pthread_cond_timedwait(clock_id: libc::clockid_t) {
    unsafe {
        let mut attr: MaybeUninit<libc::pthread_condattr_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_condattr_init(attr.as_mut_ptr()), 0);
        #[cfg(target_vendor = "apple")]
        assert_eq!(clock_id, libc::CLOCK_REALTIME); // only supported clock on macOS
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(libc::pthread_condattr_setclock(attr.as_mut_ptr(), clock_id), 0);

        let mut cond: MaybeUninit<libc::pthread_cond_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_cond_init(cond.as_mut_ptr(), attr.as_ptr()), 0);
        assert_eq!(libc::pthread_condattr_destroy(attr.as_mut_ptr()), 0);

        let mut mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;

        let before_time = Instant::now();
        let mut now_mu: MaybeUninit<libc::timespec> = MaybeUninit::uninit();
        errno_check(libc::clock_gettime(clock_id, now_mu.as_mut_ptr()));
        let now = now_mu.assume_init();
        // Waiting for 10ms.
        let new_nsec = now.tv_nsec + 10 * 1000 * 1000;
        let nanos = 1_000_000_000;
        let timeout =
            libc::timespec { tv_sec: now.tv_sec + new_nsec / nanos, tv_nsec: new_nsec % nanos };

        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        assert_eq!(
            libc::pthread_cond_timedwait(cond.as_mut_ptr(), &mut mutex as *mut _, &timeout),
            libc::ETIMEDOUT
        );
        let elapsed_time = before_time.elapsed().as_millis();
        assert!(10 <= elapsed_time && elapsed_time <= 100);

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

        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_cond_destroy(cond.as_mut_ptr()), 0);
    }
}

#[cfg(not(target_vendor = "apple"))] // macOS does not support pthread semaphores
fn test_sem_timedwait() {
    use libc_utils::errno_result;

    unsafe {
        let mut sem = std::mem::zeroed::<libc::sem_t>();
        errno_check(libc::sem_init(&raw mut sem, /* shared */ 0, /* value */ 1));

        // Test timed wait.
        let before_time = Instant::now();
        let mut now_mu: MaybeUninit<libc::timespec> = MaybeUninit::uninit();
        errno_check(libc::clock_gettime(libc::CLOCK_REALTIME, now_mu.as_mut_ptr()));
        let now = now_mu.assume_init();
        // Wait 10ms.
        let new_nsec = now.tv_nsec + 10 * 1000 * 1000;
        let nanos = 1_000_000_000;
        let timeout =
            libc::timespec { tv_sec: now.tv_sec + new_nsec / nanos, tv_nsec: new_nsec % nanos };

        // First wait decrements value to 1 immediately, 2nd wait blocks for a bit.
        errno_result(libc::sem_timedwait(&raw mut sem, &raw const timeout)).unwrap();
        let err = errno_result(libc::sem_timedwait(&raw mut sem, &raw const timeout)).unwrap_err();
        let elapsed_time = before_time.elapsed().as_millis();
        assert_eq!(err.raw_os_error().unwrap(), libc::ETIMEDOUT);
        assert!(10 <= elapsed_time && elapsed_time <= 100);
    }
}

fn main() {
    test_pthread_cond_timedwait(libc::CLOCK_REALTIME);
    if cfg!(not(target_vendor = "apple")) {
        test_pthread_cond_timedwait(libc::CLOCK_MONOTONIC);
    }

    #[cfg(not(target_vendor = "apple"))] // macOS does not support pthread semaphores
    {
        test_sem_timedwait();
    }
}
