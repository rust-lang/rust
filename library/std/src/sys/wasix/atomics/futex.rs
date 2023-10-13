use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

/// Wait for a futex_wake operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    let timeout = match timeout {
        Some(timeout) => wasi::OptionTimestamp {
            tag: wasi::OPTION_SOME.raw(),
            u: wasi::OptionTimestampU {
                some: timeout.as_nanos() as u64,
            }
        },
        None => wasi::OptionTimestamp {
            tag: wasi::OPTION_NONE.raw(),
            u: wasi::OptionTimestampU {
                none: 0,
            }
        }
    };

    unsafe {
        match wasi::futex_wait(
            futex as *const AtomicU32 as *mut u32,
            expected,
            &timeout as *const wasi::OptionTimestamp
        ) {
            Ok(wasi::BOOL_FALSE) => false,
            _ => true
        }
    }
}

/// Wake up one thread that's blocked on futex_wait on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe {
        match wasi::futex_wake(
            futex as *const AtomicU32 as *mut u32,
        ) {
            Ok(wasi::BOOL_TRUE) => true,
            Ok(_) => false,
            Err(_) => false
        }
    }
}

/// Wake up all threads that are waiting on futex_wait on this futex.
pub fn futex_wake_all(futex: &AtomicU32) -> bool {
    unsafe {
        match wasi::futex_wake_all(
            futex as *const AtomicU32 as *mut u32,
        ) {
            Ok(wasi::BOOL_TRUE) => true,
            Ok(_) => false,
            Err(_) => false
        }
    }
}
