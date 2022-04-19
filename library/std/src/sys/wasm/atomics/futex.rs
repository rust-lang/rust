use crate::arch::wasm32;
use crate::convert::TryInto;
use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

/// Wait for a futex_wake operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    let timeout = timeout.and_then(|t| t.as_nanos().try_into().ok()).unwrap_or(-1);
    unsafe {
        wasm32::memory_atomic_wait32(
            futex as *const AtomicU32 as *mut i32,
            expected as i32,
            timeout,
        ) < 2
    }
}

/// Wake up one thread that's blocked on futex_wait on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe { wasm32::memory_atomic_notify(futex as *const AtomicU32 as *mut i32, 1) > 0 }
}

/// Wake up all threads that are waiting on futex_wait on this futex.
pub fn futex_wake_all(futex: &AtomicU32) {
    unsafe {
        wasm32::memory_atomic_notify(futex as *const AtomicU32 as *mut i32, i32::MAX as u32);
    }
}
