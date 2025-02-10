#![cfg(any(
    target_os = "linux",
    target_vendor = "apple",
    target_os = "android",
    all(target_os = "emscripten", target_feature = "atomics"),
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "fuchsia",
))]

use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

/// An atomic for use as a futex that is at least 32-bits but may be larger
pub type Futex = AtomicU32;
/// Must be the underlying type of Futex
pub type Primitive = u32;

/// An atomic for use as a futex that is at least 8-bits but may be larger.
pub type SmallFutex = AtomicU32;
/// Must be the underlying type of SmallFutex
pub type SmallPrimitive = u32;

/// Waits for a `futex_wake` operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    use super::time::Timespec;
    use crate::ptr::null;
    use crate::sync::atomic::Ordering::Relaxed;

    // Calculate the timeout as an absolute timespec.
    //
    // Overflows are rounded up to an infinite timeout (None).
    let timespec = timeout
        .and_then(|d| Timespec::now(libc::CLOCK_MONOTONIC).checked_add_duration(&d))
        .and_then(|t| t.to_timespec());

    loop {
        // No need to wait if the value already changed.
        if futex.load(Relaxed) != expected {
            return true;
        }

        let r = unsafe {
            cfg_if::cfg_if! {
                if #[cfg(target_os = "freebsd")] {
                    // FreeBSD doesn't have futex(), but it has
                    // _umtx_op(UMTX_OP_WAIT_UINT_PRIVATE), which is nearly
                    // identical. It supports absolute timeouts through a flag
                    // in the _umtx_time struct.
                    let umtx_timeout = timespec.map(|t| libc::_umtx_time {
                        _timeout: t,
                        _flags: libc::UMTX_ABSTIME,
                        _clockid: libc::CLOCK_MONOTONIC as u32,
                    });
                    let umtx_timeout_ptr = umtx_timeout.as_ref().map_or(null(), |t| t as *const _);
                    let umtx_timeout_size = umtx_timeout.as_ref().map_or(0, |t| crate::mem::size_of_val(t));
                    libc::_umtx_op(
                        futex as *const AtomicU32 as *mut _,
                        libc::UMTX_OP_WAIT_UINT_PRIVATE,
                        expected as libc::c_ulong,
                        crate::ptr::without_provenance_mut(umtx_timeout_size),
                        umtx_timeout_ptr as *mut _,
                    )
                } else if #[cfg(any(target_os = "linux", target_os = "android"))] {
                    // Use FUTEX_WAIT_BITSET rather than FUTEX_WAIT to be able to give an
                    // absolute time rather than a relative time.
                    libc::syscall(
                        libc::SYS_futex,
                        futex as *const AtomicU32,
                        libc::FUTEX_WAIT_BITSET | libc::FUTEX_PRIVATE_FLAG,
                        expected,
                        timespec.as_ref().map_or(null(), |t| t as *const libc::timespec),
                        null::<u32>(), // This argument is unused for FUTEX_WAIT_BITSET.
                        !0u32,         // A full bitmask, to make it behave like a regular FUTEX_WAIT.
                    )
                } else {
                    compile_error!("unknown target_os");
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

/// Wakes up one thread that's blocked on `futex_wait` on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
///
/// On some platforms, this always returns false.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    let ptr = futex as *const AtomicU32;
    let op = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;
    unsafe { libc::syscall(libc::SYS_futex, ptr, op, 1) > 0 }
}

/// Wakes up all threads that are waiting on `futex_wait` on this futex.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wake_all(futex: &AtomicU32) {
    let ptr = futex as *const AtomicU32;
    let op = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;
    unsafe {
        libc::syscall(libc::SYS_futex, ptr, op, i32::MAX);
    }
}

// FreeBSD doesn't tell us how many threads are woken up, so this always returns false.
#[cfg(target_os = "freebsd")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    use crate::ptr::null_mut;
    unsafe {
        libc::_umtx_op(
            futex as *const AtomicU32 as *mut _,
            libc::UMTX_OP_WAKE_PRIVATE,
            1,
            null_mut(),
            null_mut(),
        )
    };
    false
}

#[cfg(target_os = "freebsd")]
pub fn futex_wake_all(futex: &AtomicU32) {
    use crate::ptr::null_mut;
    unsafe {
        libc::_umtx_op(
            futex as *const AtomicU32 as *mut _,
            libc::UMTX_OP_WAKE_PRIVATE,
            i32::MAX as libc::c_ulong,
            null_mut(),
            null_mut(),
        )
    };
}

/// With macOS version 14.4, Apple introduced a public futex API. Unfortunately,
/// our minimum supported version is 10.12, so we need a fallback API. Luckily
/// for us, the underlying syscalls have been available since exactly that
/// version, so we just use those when needed. This is private API however,
/// which means we need to take care to avoid breakage if the syscall is removed
/// and to avoid apps being rejected from the App Store. To do this, we use weak
/// linkage emulation for both the public and the private API.
///
/// See https://developer.apple.com/documentation/os/os_sync_wait_on_address?language=objc
/// for documentation of the public API and
/// https://github.com/apple-oss-distributions/xnu/blob/1031c584a5e37aff177559b9f69dbd3c8c3fd30a/bsd/sys/ulock.h#L69
/// for the header file of the private API, along with its usage in libpthread
/// https://github.com/apple-oss-distributions/libpthread/blob/d8c4e3c212553d3e0f5d76bb7d45a8acd61302dc/src/pthread_cond.c#L463
#[cfg(target_vendor = "apple")]
mod apple {
    use crate::ffi::{c_int, c_void};
    // Explicitly use `dlsym` for the private fallback API to ensure that the
    // App Store checks do not flag Apps using Rust as relying on private API.
    use crate::sys::pal::weak::{dlsym, weak};

    pub const OS_CLOCK_MACH_ABSOLUTE_TIME: u32 = 32;
    pub const OS_SYNC_WAIT_ON_ADDRESS_NONE: u32 = 0;
    pub const OS_SYNC_WAKE_BY_ADDRESS_NONE: u32 = 0;

    pub const UL_COMPARE_AND_WAIT: u32 = 1;
    pub const ULF_WAKE_ALL: u32 = 0x100;
    // The syscalls support directly returning errors instead of going through errno.
    pub const ULF_NO_ERRNO: u32 = 0x1000000;

    // These functions appeared with macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, visionOS 1.1.
    weak! {
        pub fn os_sync_wait_on_address(*mut c_void, u64, usize, u32) -> c_int
    }

    weak! {
        pub fn os_sync_wait_on_address_with_timeout(*mut c_void, u64, usize, u32, u32, u64) -> c_int
    }

    weak! {
        pub fn os_sync_wake_by_address_any(*mut c_void, usize, u32) -> c_int
    }

    weak! {
        pub fn os_sync_wake_by_address_all(*mut c_void, usize, u32) -> c_int
    }

    // This syscall appeared with macOS 11.0.
    // It is used to support nanosecond precision for timeouts, among other features.
    dlsym! {
        pub fn __ulock_wait2(u32, *mut c_void, u64, u64, u64) -> c_int
    }

    // These syscalls appeared with macOS 10.12.
    dlsym! {
        pub fn __ulock_wait(u32, *mut c_void, u64, u32) -> c_int
    }

    dlsym! {
        pub fn __ulock_wake(u32, *mut c_void, u64) -> c_int
    }
}

#[cfg(target_vendor = "apple")]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    use apple::*;

    use crate::mem::size_of;

    let addr = futex.as_ptr().cast();
    let value = expected as u64;
    let size = size_of::<u32>();
    if let Some(timeout) = timeout {
        let timeout_ns = timeout.as_nanos().clamp(1, u64::MAX as u128) as u64;
        let timeout_ms = timeout.as_micros().clamp(1, u32::MAX as u128) as u32;

        if let Some(wait) = os_sync_wait_on_address_with_timeout.get() {
            let r = unsafe {
                wait(
                    addr,
                    value,
                    size,
                    OS_SYNC_WAIT_ON_ADDRESS_NONE,
                    OS_CLOCK_MACH_ABSOLUTE_TIME,
                    timeout_ns,
                )
            };

            // We promote spurious wakeups (reported as EINTR) to normal ones for
            // simplicity.
            r != -1 || super::os::errno() != libc::ETIMEDOUT
        } else if let Some(wait) = __ulock_wait2.get() {
            unsafe {
                wait(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, value, timeout_ns, 0)
                    != -libc::ETIMEDOUT
            }
        } else if let Some(wait) = __ulock_wait.get() {
            unsafe {
                wait(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, value, timeout_ms)
                    != -libc::ETIMEDOUT
            }
        } else {
            panic!("your system is below the minimum supported version of Rust");
        }
    } else {
        if let Some(wait) = os_sync_wait_on_address.get() {
            unsafe {
                wait(addr, value, size, OS_SYNC_WAIT_ON_ADDRESS_NONE);
            }
        } else if let Some(wait) = __ulock_wait.get() {
            unsafe {
                wait(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, value, 0);
            }
        } else {
            panic!("your system is below the minimum supported version of Rust");
        }
        true
    }
}

#[cfg(target_vendor = "apple")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    use apple::*;

    use crate::io::Error;
    use crate::mem::size_of;

    let addr = futex.as_ptr().cast();
    if let Some(wake) = os_sync_wake_by_address_any.get() {
        unsafe { wake(addr, size_of::<u32>(), OS_SYNC_WAKE_BY_ADDRESS_NONE) == 0 }
    } else if let Some(wake) = __ulock_wake.get() {
        // Judging by its use in pthreads, __ulock_wake can get interrupted, so
        // retry until either waking up a waiter or failing because there are no
        // waiters (ENOENT).
        loop {
            let r = unsafe { wake(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, 0) };

            if r >= 0 {
                return true;
            } else {
                match -r {
                    libc::ENOENT => return false,
                    libc::EINTR => continue,
                    err => panic!("__ulock_wake failed: {}", Error::from_raw_os_error(err)),
                }
            }
        }
    } else {
        panic!("your system is below the minimum supported version of Rust");
    }
}

#[cfg(target_vendor = "apple")]
pub fn futex_wake_all(futex: &AtomicU32) {
    use apple::*;

    use crate::io::Error;
    use crate::mem::size_of;

    let addr = futex.as_ptr().cast();

    if let Some(wake) = os_sync_wake_by_address_all.get() {
        unsafe {
            wake(addr, size_of::<u32>(), OS_SYNC_WAKE_BY_ADDRESS_NONE);
        }
    } else if let Some(wake) = __ulock_wake.get() {
        loop {
            let r = unsafe { wake(UL_COMPARE_AND_WAIT | ULF_WAKE_ALL | ULF_NO_ERRNO, addr, 0) };

            if r >= 0 {
                return;
            } else {
                match -r {
                    libc::ENOENT => return,
                    libc::EINTR => continue,
                    err => panic!("__ulock_wake failed: {}", Error::from_raw_os_error(err)),
                }
            }
        }
    } else {
        panic!("your system is below the minimum supported version of Rust");
    }
}

#[cfg(target_os = "openbsd")]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    use super::time::Timespec;
    use crate::ptr::{null, null_mut};

    // Overflows are rounded up to an infinite timeout (None).
    let timespec = timeout
        .and_then(|d| Timespec::zero().checked_add_duration(&d))
        .and_then(|t| t.to_timespec());

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
    // A timeout of 0 means infinite.
    // We round smaller timeouts up to 1 millisecond.
    // Overflows are rounded up to an infinite timeout.
    let timeout_ms =
        timeout.and_then(|d| Some(i32::try_from(d.as_millis()).ok()?.max(1))).unwrap_or(0);

    let r = unsafe {
        libc::umtx_sleep(futex as *const AtomicU32 as *const i32, expected as i32, timeout_ms)
    };

    r == 0 || super::os::errno() != libc::ETIMEDOUT
}

// DragonflyBSD doesn't tell us how many threads are woken up, so this always returns false.
#[cfg(target_os = "dragonfly")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe { libc::umtx_wakeup(futex as *const AtomicU32 as *const i32, 1) };
    false
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

#[cfg(target_os = "fuchsia")]
pub mod zircon {
    pub type zx_futex_t = crate::sync::atomic::AtomicU32;
    pub type zx_handle_t = u32;
    pub type zx_status_t = i32;
    pub type zx_time_t = i64;

    pub const ZX_HANDLE_INVALID: zx_handle_t = 0;

    pub const ZX_TIME_INFINITE: zx_time_t = zx_time_t::MAX;

    pub const ZX_OK: zx_status_t = 0;
    pub const ZX_ERR_INVALID_ARGS: zx_status_t = -10;
    pub const ZX_ERR_BAD_HANDLE: zx_status_t = -11;
    pub const ZX_ERR_WRONG_TYPE: zx_status_t = -12;
    pub const ZX_ERR_BAD_STATE: zx_status_t = -20;
    pub const ZX_ERR_TIMED_OUT: zx_status_t = -21;

    extern "C" {
        pub fn zx_clock_get_monotonic() -> zx_time_t;
        pub fn zx_futex_wait(
            value_ptr: *const zx_futex_t,
            current_value: zx_futex_t,
            new_futex_owner: zx_handle_t,
            deadline: zx_time_t,
        ) -> zx_status_t;
        pub fn zx_futex_wake(value_ptr: *const zx_futex_t, wake_count: u32) -> zx_status_t;
        pub fn zx_futex_wake_single_owner(value_ptr: *const zx_futex_t) -> zx_status_t;
        pub fn zx_thread_self() -> zx_handle_t;
    }
}

#[cfg(target_os = "fuchsia")]
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    // Sleep forever if the timeout is longer than fits in a i64.
    let deadline = timeout
        .and_then(|d| {
            i64::try_from(d.as_nanos())
                .ok()?
                .checked_add(unsafe { zircon::zx_clock_get_monotonic() })
        })
        .unwrap_or(zircon::ZX_TIME_INFINITE);

    unsafe {
        zircon::zx_futex_wait(futex, AtomicU32::new(expected), zircon::ZX_HANDLE_INVALID, deadline)
            != zircon::ZX_ERR_TIMED_OUT
    }
}

// Fuchsia doesn't tell us how many threads are woken up, so this always returns false.
#[cfg(target_os = "fuchsia")]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe { zircon::zx_futex_wake(futex, 1) };
    false
}

#[cfg(target_os = "fuchsia")]
pub fn futex_wake_all(futex: &AtomicU32) {
    unsafe { zircon::zx_futex_wake(futex, u32::MAX) };
}
