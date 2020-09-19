#![cfg(any(target_os = "linux", target_os = "android"))]

use crate::sync::atomic::AtomicI32;
use crate::time::Duration;

pub fn futex_wait(futex: &AtomicI32, expected: i32, timeout: Option<Duration>) {
    let timespec;
    let timespec_ptr = match timeout {
        Some(timeout) => {
            timespec = libc::timespec {
                tv_sec: timeout.as_secs() as _,
                tv_nsec: timeout.subsec_nanos() as _,
            };
            &timespec as *const libc::timespec
        }
        None => crate::ptr::null(),
    };
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicI32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            timespec_ptr,
        );
    }
}

pub fn futex_wake(futex: &AtomicI32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicI32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1,
        );
    }
}
