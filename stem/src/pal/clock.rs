//! Clock and time platform abstraction.
//!
//! Provides monotonic time access via the platform syscall layer.
//! All time values are in nanoseconds.

use crate::syscall;
use abi::time::ClockId;

/// Returns raw monotonic nanoseconds since boot.
///
/// This is the fundamental time primitive. The value is guaranteed to be
/// monotonically increasing and unaffected by wall-clock adjustments.
#[inline]
pub fn monotonic_ns() -> u64 {
    syscall::monotonic_ns()
}

/// Sleep for the specified number of nanoseconds.
///
/// The actual sleep duration may be longer due to scheduling granularity.
#[inline]
pub fn sleep_ns(ns: u64) {
    syscall::sleep_ns(ns);
}

/// Returns the current Unix time in nanoseconds.
///
/// Returns 0 if the system clock is not yet anchored to real time.
pub fn unix_time_ns() -> u64 {
    syscall::time_now(ClockId::Realtime)
        .ok()
        .and_then(|spec| spec.as_nanos())
        .unwrap_or(0)
}
