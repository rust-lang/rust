#![cfg(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics")
))]

use crate::sync::atomic::AtomicI32;
use crate::time::Duration;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wait(futex: &AtomicI32, expected: i32, timeout: Option<Duration>) -> bool {
    use super::time::Instant;
    use crate::ptr::null;
    use crate::sync::atomic::Ordering::Relaxed;

    // Calculate the timeout as an absolute timespec.
    let timespec =
        timeout.and_then(|d| Some(Instant::now().checked_add_duration(&d)?.as_timespec()));

    loop {
        // No need to wait if the value already changed.
        if futex.load(Relaxed) != expected {
            return true;
        }

        // Use FUTEX_WAIT_BITSET rather than FUTEX_WAIT to be able to give an
        // absolute time rather than a relative time.
        let r = unsafe {
            libc::syscall(
                libc::SYS_futex,
                futex as *const AtomicI32,
                libc::FUTEX_WAIT_BITSET | libc::FUTEX_PRIVATE_FLAG,
                expected,
                timespec.as_ref().map_or(null(), |d| d as *const libc::timespec),
                null::<u32>(), // This argument is unused for FUTEX_WAIT_BITSET.
                !0u32,         // A full bitmask, to make it behave like a regular FUTEX_WAIT.
            )
        };

        match (r < 0).then(super::os::errno) {
            Some(libc::ETIMEDOUT) => return false,
            Some(libc::EINTR) => continue,
            _ => return true,
        }
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

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wake_all(futex: &AtomicI32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicI32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
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
