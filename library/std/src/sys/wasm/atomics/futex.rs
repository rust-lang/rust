use crate::arch::wasm32;
use crate::convert::TryInto;
use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) {
    let timeout = timeout.and_then(|t| t.as_nanos().try_into().ok()).unwrap_or(-1);
    unsafe {
        wasm32::memory_atomic_wait32(
            futex as *const AtomicU32 as *mut i32,
            expected as i32,
            timeout,
        );
    }
}

pub fn futex_wake(futex: &AtomicU32) {
    unsafe {
        wasm32::memory_atomic_notify(futex as *const AtomicU32 as *mut i32, 1);
    }
}
