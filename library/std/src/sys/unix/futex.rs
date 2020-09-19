#![cfg(any(target_os = "linux", target_os = "android"))]

use crate::convert::TryInto;
use crate::ptr::null;
use crate::sync::atomic::AtomicI32;
use crate::time::Duration;

pub fn futex_wait(futex: &AtomicI32, expected: i32, timeout: Option<Duration>) {
    let timespec = timeout.and_then(|d| {
        Some(libc::timespec {
            // Sleep forever if the timeout is longer than fits in a timespec.
            tv_sec: d.as_secs().try_into().ok()?,
            // This conversion never truncates, as subsec_nanos is always <1e9.
            tv_nsec: d.subsec_nanos() as _,
        })
    });
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicI32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            timespec.as_ref().map_or(null(), |d| d as *const libc::timespec),
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
