use core::sync::atomic::Ordering;

use crate::sync::atomic::Atomic;
use crate::time::{Duration, Instant};

/// An atomic for use as a futex that is at least 32-bits but may be larger
pub type Futex = Atomic<Primitive>;
/// Must be the underlying type of Futex
pub type Primitive = u32;

/// An atomic for use as a futex that is at least 8-bits but may be larger.
pub type SmallFutex = Atomic<SmallPrimitive>;
/// Must be the underlying type of SmallFutex
pub type SmallPrimitive = u32;

/// Wait for a futex_wake operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
pub fn futex_wait(futex: &Atomic<u32>, expected: u32, timeout: Option<Duration>) -> bool {
    if let Some(timeout) = timeout {
        let begin = Instant::now();

        while futex.load(Ordering::Acquire) == expected {
            if begin.elapsed() >= timeout {
                return false;
            }

            // Wait for an ISR or Simple Task to wake.
            crate::thread::yield_now();
        }
    } else {
        while futex.load(Ordering::Acquire) == expected {
            crate::thread::yield_now();
        }
    }

    true
}

/// Wakes up one thread that's blocked on `futex_wait` on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
pub fn futex_wake(_futex: &Atomic<u32>) -> bool {
    // This matches the behavior of FreeBSD/DragonFlyBSD which also always return false here.
    false
}

/// Wakes up all threads that are waiting on `futex_wait` on this futex.
pub fn futex_wake_all(_futex: &Atomic<u32>) {
    // The futex_wait will wake itself up whenever the futex is modified, so this can
    // stay a no-op.
}
