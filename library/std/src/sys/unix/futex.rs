#![cfg(any(
    target_os = "linux",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics"),
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
))]

use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

#[cfg(target_os = "netbsd")]
pub const SYS___futex: i32 = 166;

/// Wait for a futex_wake operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "netbsd"))]
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
            cfg_if::cfg_if! {
                if #[cfg(target_os = "netbsd")] {
                    // Netbsd's futex syscall takes addr2 and val2 as separate arguments.
                    // (Both are unused for FUTEX_WAIT[_BITSET].)
                    libc::syscall(
                        SYS___futex,
                        futex as *const AtomicU32,
                        libc::FUTEX_WAIT_BITSET | libc::FUTEX_PRIVATE_FLAG,
                        expected,
                        timespec.as_ref().map_or(null(), |t| &t.t as *const libc::timespec),
                        null::<u32>(), // addr2: This argument is unused for FUTEX_WAIT_BITSET.
                        0,             // val2: This argument is unused for FUTEX_WAIT_BITSET.
                        !0u32,         // val3 / bitmask: A full bitmask, to make it behave like a regular FUTEX_WAIT.
                    )
                } else {
                    libc::syscall(
                        libc::SYS_futex,
                        futex as *const AtomicU32,
                        libc::FUTEX_WAIT_BITSET | libc::FUTEX_PRIVATE_FLAG,
                        expected,
                        timespec.as_ref().map_or(null(), |t| &t.t as *const libc::timespec),
                        null::<u32>(), // This argument is unused for FUTEX_WAIT_BITSET.
                        !0u32,         // A full bitmask, to make it behave like a regular FUTEX_WAIT.
                    )
                }
            }
        };

        match (r < 0).then(super::os::errno) {
            Some(libc::ETIMEDOUT) => return false,
            Some(libc::EINTR) => continue,
            _ => return true,
        }
    }
}

/// Wake up one thread that's blocked on futex_wait on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "netbsd"))]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    let ptr = futex as *const AtomicU32;
    let op = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_os = "netbsd")] {
                libc::syscall(SYS___futex, ptr, op, 1) > 0
            } else {
                libc::syscall(libc::SYS_futex, ptr, op, 1) > 0
            }
        }
    }
}

/// Wake up all threads that are waiting on futex_wait on this futex.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "netbsd"))]
pub fn futex_wake_all(futex: &AtomicU32) {
    let ptr = futex as *const AtomicU32;
    let op = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;
    unsafe {
        cfg_if::cfg_if! {
            if #[cfg(target_os = "netbsd")] {
                libc::syscall(SYS___futex, ptr, op, i32::MAX);
            } else {
                libc::syscall(libc::SYS_futex, ptr, op, i32::MAX);
            }
        }
    }
}

#[cfg(target_os = "openbsd")]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    use crate::convert::TryInto;
    use crate::ptr::{null, null_mut};
    let timespec = timeout.and_then(|d| {
        Some(libc::timespec {
            // Sleep forever if the timeout is longer than fits in a timespec.
            tv_sec: d.as_secs().try_into().ok()?,
            // This conversion never truncates, as subsec_nanos is always <1e9.
            tv_nsec: d.subsec_nanos() as _,
        })
    });

    let r = unsafe {
        libc::futex(
            futex as *const AtomicU32 as *mut u32,
            libc::FUTEX_WAIT,
            expected as i32,
            timespec.as_ref().map_or(null(), |t| t as *const libc::timespec),
            null_mut(),
        )
    };

    r == 0 || super::os::errno() != libc::ETIMEDOUT
}

#[cfg(target_os = "openbsd")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    use crate::ptr::{null, null_mut};
    unsafe {
        libc::futex(futex as *const AtomicU32 as *mut u32, libc::FUTEX_WAKE, 1, null(), null_mut())
            > 0
    }
}

#[cfg(target_os = "openbsd")]
pub fn futex_wake_all(futex: &AtomicU32) {
    use crate::ptr::{null, null_mut};
    unsafe {
        libc::futex(
            futex as *const AtomicU32 as *mut u32,
            libc::FUTEX_WAKE,
            i32::MAX,
            null(),
            null_mut(),
        );
    }
}

#[cfg(target_os = "dragonfly")]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    use crate::convert::TryFrom;
    let r = unsafe {
        libc::umtx_sleep(
            futex as *const AtomicU32 as *const i32,
            expected as i32,
            // A timeout of 0 means infinite, so we round smaller timeouts up to 1 millisecond.
            // Timeouts larger than i32::MAX milliseconds saturate.
            timeout.map_or(0, |d| {
                i32::try_from(d.as_millis()).map_or(i32::MAX, |millis| millis.max(1))
            }),
        )
    };

    r == 0 || super::os::errno() != libc::ETIMEDOUT
}

// DragonflyBSD doesn't tell us how many threads are woken up, so this doesn't return a bool.
#[cfg(target_os = "dragonfly")]
pub fn futex_wake(futex: &AtomicU32) {
    unsafe { libc::umtx_wakeup(futex as *const AtomicU32 as *const i32, 1) };
}

#[cfg(target_os = "dragonfly")]
pub fn futex_wake_all(futex: &AtomicU32) {
    unsafe { libc::umtx_wakeup(futex as *const AtomicU32 as *const i32, i32::MAX) };
}

#[cfg(target_os = "emscripten")]
extern "C" {
    fn emscripten_futex_wake(addr: *const AtomicU32, count: libc::c_int) -> libc::c_int;
    fn emscripten_futex_wait(
        addr: *const AtomicU32,
        val: libc::c_uint,
        max_wait_ms: libc::c_double,
    ) -> libc::c_int;
}

#[cfg(target_os = "emscripten")]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    unsafe {
        emscripten_futex_wait(
            futex,
            expected,
            timeout.map_or(f64::INFINITY, |d| d.as_secs_f64() * 1000.0),
        ) != -libc::ETIMEDOUT
    }
}

#[cfg(target_os = "emscripten")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe { emscripten_futex_wake(futex, 1) > 0 }
}

#[cfg(target_os = "emscripten")]
pub fn futex_wake_all(futex: &AtomicU32) {
    unsafe { emscripten_futex_wake(futex, i32::MAX) };
}
