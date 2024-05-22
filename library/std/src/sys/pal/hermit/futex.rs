use super::hermit_abi;
use crate::ptr::null;
use crate::sync::atomic::AtomicU32;
use crate::time::Duration;

pub fn futex_wait(futex: &AtomicU32, expected: u32, timeout: Option<Duration>) -> bool {
    // Calculate the timeout as a relative timespec.
    //
    // Overflows are rounded up to an infinite timeout (None).
    let timespec = timeout.and_then(|dur| {
        Some(hermit_abi::timespec {
            tv_sec: dur.as_secs().try_into().ok()?,
            tv_nsec: dur.subsec_nanos().into(),
        })
    });

    let r = unsafe {
        hermit_abi::futex_wait(
            futex.as_ptr(),
            expected,
            timespec.as_ref().map_or(null(), |t| t as *const hermit_abi::timespec),
            hermit_abi::FUTEX_RELATIVE_TIMEOUT,
        )
    };

    r != -hermit_abi::errno::ETIMEDOUT
}

#[inline]
pub fn futex_wake(futex: &AtomicU32) -> bool {
    unsafe { hermit_abi::futex_wake(futex.as_ptr(), 1) > 0 }
}

#[inline]
pub fn futex_wake_all(futex: &AtomicU32) {
    unsafe {
        hermit_abi::futex_wake(futex.as_ptr(), i32::MAX);
    }
}
