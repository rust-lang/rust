use crate::sync::atomic::Atomic;
use crate::sync::atomic::Ordering::Relaxed;
use crate::time::Duration;

/// An atomic for use as a futex that is at least 32-bits but may be larger.
pub type Futex = Atomic<Primitive>;
/// Must be the underlying type of Futex.
pub type Primitive = u32;

/// An atomic for use as a futex that is at least 8-bits but may be larger.
pub type SmallFutex = Atomic<SmallPrimitive>;
/// Must be the underlying type of SmallFutex.
pub type SmallPrimitive = u32;

// Syscall numbers from ThingOS ABI (abi/src/numbers.rs).
const SYS_FUTEX_WAIT: u32 = 0x1300;
const SYS_FUTEX_WAKE: u32 = 0x1301;

const EINTR: i32 = 4;
const ETIMEDOUT: i32 = 110;

#[inline]
fn duration_to_timeout_ns(timeout: Option<Duration>) -> u64 {
    match timeout {
        None => 0,
        Some(d) => {
            let ns = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
            // Kernel uses 0 for an infinite wait; clamp zero-duration waits to 1ns.
            ns.max(1)
        }
    }
}

/// Waits for a `futex_wake` operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
/// Returns false on timeout and true in all other cases.
pub fn futex_wait(futex: &Atomic<u32>, expected: u32, timeout: Option<Duration>) -> bool {
    loop {
        // Avoid syscall if the futex value no longer matches.
        if futex.load(Relaxed) != expected {
            return true;
        }

        let ret = unsafe {
            crate::sys::pal::raw_syscall6(
                SYS_FUTEX_WAIT,
                futex as *const Atomic<u32> as usize,
                expected as usize,
                duration_to_timeout_ns(timeout) as usize,
                0,
                0,
                0,
            )
        };

        if ret >= 0 {
            return true;
        }

        let errno = (-ret) as i32;
        if errno == EINTR {
            continue;
        }
        if errno == ETIMEDOUT {
            return false;
        }
        return true;
    }
}

/// Wakes up one thread waiting on this futex.
///
/// Returns true if this likely woke a waiter, false otherwise.
pub fn futex_wake(futex: &Atomic<u32>) -> bool {
    let ret = unsafe {
        crate::sys::pal::raw_syscall6(
            SYS_FUTEX_WAKE,
            futex as *const Atomic<u32> as usize,
            1,
            0,
            0,
            0,
            0,
        )
    };
    ret > 0
}

/// Wakes all threads waiting on this futex.
pub fn futex_wake_all(futex: &Atomic<u32>) {
    unsafe {
        crate::sys::pal::raw_syscall6(
            SYS_FUTEX_WAKE,
            futex as *const Atomic<u32> as usize,
            i32::MAX as usize,
            0,
            0,
            0,
            0,
        );
    }
}
