//@only-target: apple # `pthread_cond_timedwait_relative_np` is a non-standard extension

use std::time::Instant;

// FIXME: remove once this is in libc.
mod libc {
    pub use ::libc::*;
    unsafe extern "C" {
        pub unsafe fn pthread_cond_timedwait_relative_np(
            cond: *mut libc::pthread_cond_t,
            lock: *mut libc::pthread_mutex_t,
            timeout: *const libc::timespec,
        ) -> libc::c_int;
    }
}

fn main() {
    unsafe {
        let mut mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;
        let mut cond: libc::pthread_cond_t = libc::PTHREAD_COND_INITIALIZER;

        // Wait for 100 ms.
        let timeout = libc::timespec { tv_sec: 0, tv_nsec: 100_000_000 };

        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);

        let current_time = Instant::now();
        assert_eq!(
            libc::pthread_cond_timedwait_relative_np(&mut cond, &mut mutex, &timeout),
            libc::ETIMEDOUT
        );
        let elapsed_time = current_time.elapsed().as_millis();
        // This is actually deterministic (since isolation remains enabled),
        // but can change slightly with Rust updates.
        assert!(90 <= elapsed_time && elapsed_time <= 110);

        assert_eq!(libc::pthread_mutex_unlock(&mut mutex), 0);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex), 0);
        assert_eq!(libc::pthread_cond_destroy(&mut cond), 0);
    }
}
