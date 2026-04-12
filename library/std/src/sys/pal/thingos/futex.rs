//! ThingOS futex-based synchronisation primitives.
//!
//! These are used by the futex-based implementations of `Mutex`, `Condvar`,
//! `RwLock`, and `Parker` (thread parking).

use super::common::{FUTEX_WAIT, FUTEX_WAKE, SYS_FUTEX, Timespec, raw_syscall6};
use crate::sync::atomic::Atomic;
use crate::time::Duration;

/// An atomic for use as a futex that is at least 32 bits but may be larger.
pub type Futex = Atomic<Primitive>;
/// The underlying integer type of [`Futex`].
pub type Primitive = u32;

/// An atomic for use as a futex that is at least 8 bits but may be larger.
pub type SmallFutex = Atomic<SmallPrimitive>;
/// The underlying integer type of [`SmallFutex`].
pub type SmallPrimitive = u32;

/// Block the current thread until either:
/// - the futex word is no longer equal to `expected`, or
/// - `timeout` elapses.
///
/// Returns `false` on timeout, `true` otherwise.
pub fn futex_wait(futex: &Atomic<u32>, expected: u32, timeout: Option<Duration>) -> bool {
    let timespec = timeout.and_then(|dur| {
        let secs = dur.as_secs().try_into().ok()?;
        let nsec = dur.subsec_nanos() as i64;
        Some(Timespec { tv_sec: secs, tv_nsec: nsec })
    });

    let ts_ptr: u64 = match timespec.as_ref() {
        Some(ts) => ts as *const Timespec as u64,
        None => 0,
    };

    let ret = unsafe {
        raw_syscall6(
            SYS_FUTEX,
            futex.as_ptr() as u64,
            FUTEX_WAIT,
            expected as u64,
            ts_ptr,
            0,
            0,
        )
    };

    // ETIMEDOUT = 110 on ThingOS; a negative return means -errno.
    ret != -110
}

/// Wake one thread waiting on `futex`.
///
/// Returns `true` if at least one thread was woken.
#[inline]
pub fn futex_wake(futex: &Atomic<u32>) -> bool {
    let ret = unsafe {
        raw_syscall6(SYS_FUTEX, futex.as_ptr() as u64, FUTEX_WAKE, 1, 0, 0, 0)
    };
    ret > 0
}

/// Wake all threads waiting on `futex`.
#[inline]
pub fn futex_wake_all(futex: &Atomic<u32>) {
    unsafe {
        raw_syscall6(SYS_FUTEX, futex.as_ptr() as u64, FUTEX_WAKE, i32::MAX as u64, 0, 0, 0);
    }
}
