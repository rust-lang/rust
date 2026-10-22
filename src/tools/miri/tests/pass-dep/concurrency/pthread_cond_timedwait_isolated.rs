//@ignore-target: windows # No pthreads on Windows
//@ignore-target: apple # pthread_condattr_setclock is not supported on MacOS.

/// Test that conditional variable timeouts are working properly
/// with monotonic clocks even under isolation.
use std::mem::MaybeUninit;
use std::time::Instant;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::errno_check;

fn test_timed_wait_timeout(clock_id: i32) {
    unsafe {
        let mut attr: MaybeUninit<libc::pthread_condattr_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_condattr_init(attr.as_mut_ptr()), 0);
        assert_eq!(libc::pthread_condattr_setclock(attr.as_mut_ptr(), clock_id), 0);

        let mut cond: MaybeUninit<libc::pthread_cond_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_cond_init(cond.as_mut_ptr(), attr.as_ptr()), 0);
        assert_eq!(libc::pthread_condattr_destroy(attr.as_mut_ptr()), 0);

        let mut mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;

        let before_time = Instant::now();
        let mut now_mu: MaybeUninit<libc::timespec> = MaybeUninit::uninit();
        errno_check(libc::clock_gettime(clock_id, now_mu.as_mut_ptr()));
        let now = now_mu.assume_init();
        // Waiting for an hour. We anyway don't really sleep for that time.
        let timeout = libc::timespec { tv_sec: now.tv_sec + 3600, tv_nsec: now.tv_nsec };

        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        assert_eq!(
            libc::pthread_cond_timedwait(cond.as_mut_ptr(), &mut mutex as *mut _, &timeout),
            libc::ETIMEDOUT
        );
        let elapsed_time = before_time.elapsed().as_secs();
        assert!(3600 <= elapsed_time && elapsed_time <= 3601);

        // Test calling `pthread_cond_timedwait` again with an already elapsed timeout.
        assert_eq!(
            libc::pthread_cond_timedwait(cond.as_mut_ptr(), &mut mutex as *mut _, &timeout),
            libc::ETIMEDOUT
        );

        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_cond_destroy(cond.as_mut_ptr()), 0);
    }
}

fn main() {
    test_timed_wait_timeout(libc::CLOCK_MONOTONIC);
}
