use crate::arch::wasm32;
use crate::convert::TryInto;
use crate::sync::atomic::AtomicI32;
use crate::time::Duration;

pub fn futex_wait(futex: &AtomicI32, expected: i32, timeout: Option<Duration>) {
    let timeout = timeout.and_then(|t| t.as_nanos().try_into().ok()).unwrap_or(-1);
    unsafe {
        wasm32::memory_atomic_wait32(futex as *const AtomicI32 as *mut i32, expected, timeout);
    }
}

pub fn futex_wake(futex: &AtomicI32) {
    unsafe {
        wasm32::memory_atomic_notify(futex as *const AtomicI32 as *mut i32, 1);
    }
}
