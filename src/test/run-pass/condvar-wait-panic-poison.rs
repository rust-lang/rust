// Test that panicking inside `Condvar::wait` doesn't poison the `Mutex`.
//
// Various platforms may trigger a panic while a thread is blocked, due to an
// error condition. It can be tricky to trigger such a panic. The test here
// shims `pthread_cond_timedwait` on Unix-like systems to trigger an assertion.
// If at some point in the future, the assertion is changed or removed so that
// the panic no longer happens, that doesn't mean this test should be removed.
// Instead, another way should be found to trigger a panic inside
// `Condvar::wait`.

// only-unix

#![feature(rustc_private)]

extern crate libc;

#[no_mangle]
pub unsafe extern "C" fn pthread_cond_timedwait(
    _cond: *mut libc::pthread_cond_t,
    _mutex: *mut libc::pthread_mutex_t,
    _abstime: *const libc::timespec
) -> libc::c_int {
    // Linux `man pthread_cond_timedwait` says EINTR may be returned
    *libc::__errno_location() = libc::EINTR;
    return 1;
}

use std::sync::{Condvar, Mutex};

fn main() {
    let m = Mutex::new(());

    std::panic::catch_unwind(|| {
        let one_ms = std::time::Duration::from_millis(2000);
        Condvar::new().wait_timeout(m.lock().unwrap(), one_ms).unwrap();
    }).expect_err("Condvar::wait should panic");

    let _ = m.lock().expect("Mutex mustn't be poisoned");
}
