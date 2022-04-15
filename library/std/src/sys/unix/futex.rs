#![cfg(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics")
))]

use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

/// Wait for a futex_wake operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    use super::time::Timespec;
    use crate::ptr::null;
    use crate::sync::atomic::Ordering::Relaxed;

    // Calculate the timeout as an absolute timespec.
    //
    // Overflows are rounded up to an infinite timeout (None).
    let timespec =
        timeout.and_then(|d| Some(Timespec::now(libc::CLOCK_MONOTONIC).checked_add_duration(&d)?));

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
                futex as *const AtomicU32,
                libc::FUTEX_WAIT_BITSET | libc::FUTEX_PRIVATE_FLAG,
                expected,
                timespec.as_ref().map_or(null(), |t| &t.t as *const libc::timespec),
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
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) {
    extern "C" {
        fn emscripten_futex_wait(
            addr: *const AtomicU32,
            val: libc::c_uint,
            max_wait_ms: libc::c_double,
        ) -> libc::c_int;
    }

    unsafe {
        emscripten_futex_wait(
            futex,
            expected,
            timeout.map_or(crate::f64::INFINITY, |d| d.as_secs_f64() * 1000.0),
        );
    }
}

/// Wake up one thread that's blocked on futex_wait on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            1,
        ) > 0
    }
}

/// Wake up all threads that are waiting on futex_wait on this futex.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wake_all(futex: &AtomicU32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
        );
    }
}

#[cfg(target_os = "emscripten")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    extern "C" {
        fn emscripten_futex_wake(addr: *const AtomicU32, count: libc::c_int) -> libc::c_int;
    }

    unsafe { emscripten_futex_wake(futex, 1) > 0 }
}
