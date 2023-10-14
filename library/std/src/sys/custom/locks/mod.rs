/// Futex-based locking
///
/// Platforms can supply hooks for the following operations:
/// - futex_wait
/// - futex_wake
/// - futex_wake_all
/// - kernel_hold_interrupts
/// - kernel_release_interrupts
use core::sync::atomic::AtomicU32;
use core::time::Duration;

use crate::os::custom::futex::IMPL;

mod condvar;
mod mutex;
mod rwlock;
pub use condvar::Condvar;
pub use mutex::Mutex;
pub use rwlock::RwLock;

/// Wait for a futex_wake operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    let reader = IMPL.read().expect("poisoned lock");
    if let Some(futex_impl) = reader.as_ref() {
        futex_impl.futex_wait(futex, expected, timeout)
    } else {
        // as this does nothing, the caller will immediately
        // try to lock again: this is spinlock behavior.
        false
    }
}

/// Wake up one thread that's blocked on futex_wait on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
///
/// On some platforms, this always returns false.
pub fn futex_wake(futex: &AtomicU32) -> bool {
    let reader = IMPL.read().expect("poisoned lock");
    if let Some(futex_impl) = reader.as_ref() {
        futex_impl.futex_wake(futex)
    } else {
        // nothing to do + no-one can be woken up
        false
    }
}

/// Wake up all threads that are waiting on futex_wait on this futex.
pub fn futex_wake_all(futex: &AtomicU32) {
    let reader = IMPL.read().expect("poisoned lock");
    if let Some(futex_impl) = reader.as_ref() {
        futex_impl.futex_wake_all(futex)
    } else {
        // nothing to do
    }
}
