#![cfg(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics")
))]

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::convert::TryInto;
#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::ptr::null;
use crate::sync::atomic::AtomicI32;
use crate::time::Duration;

#[cfg(any(target_os = "linux", target_os = "android"))]
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

#[cfg(target_os = "emscripten")]
pub fn futex_wait(futex: &AtomicI32, expected: i32, timeout: Option<Duration>) {
    extern "C" {
        fn emscripten_futex_wait(
            addr: *const AtomicI32,
            val: libc::c_uint,
            max_wait_ms: libc::c_double,
        ) -> libc::c_int;
    }

    unsafe {
        emscripten_futex_wait(
            futex as *const AtomicI32,
            // `val` is declared unsigned to match the Emscripten headers, but since it's used as
            // an opaque value, we can ignore the meaning of signed vs. unsigned and cast here.
            expected as libc::c_uint,
            timeout.map_or(crate::f64::INFINITY, |d| d.as_secs_f64() * 1000.0),
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
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

#[cfg(target_os = "emscripten")]
pub fn futex_wake(futex: &AtomicI32) {
    extern "C" {
        fn emscripten_futex_wake(addr: *const AtomicI32, count: libc::c_int) -> libc::c_int;
    }

    unsafe {
        emscripten_futex_wake(futex as *const AtomicI32, 1);
    }
}
