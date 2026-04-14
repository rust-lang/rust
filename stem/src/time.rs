//! Time primitives for userspace.
//!
//! All time in Thing-OS derives from a single monotonic timebase.
//! Use `now()` to get the current `Instant`.

use crate::pal;

// Re-export the ABI types for convenience
pub use abi::time::{ClockId, TimeSpec};
pub use abi::types::instant::{Duration, Instant};

/// Returns the current monotonic instant.
///
/// This is the primary way to get the current time in userspace.
/// The returned `Instant` is guaranteed to be monotonically increasing.
#[inline]
pub fn now() -> Instant {
    Instant::from_nanos(pal::clock::monotonic_ns())
}

/// Returns the current Unix time in nanoseconds.
///
/// Returns 0 if the system clock is not yet anchored.
pub fn now_unix_nanos() -> u64 {
    pal::clock::unix_time_ns()
}

/// Returns the current clock value for the requested clock domain.
pub fn clock_now(clock_id: ClockId) -> Option<TimeSpec> {
    crate::syscall::time_now(clock_id).ok()
}

/// Returns the current Unix time in seconds.
///
/// Returns 0 if the system clock is not yet anchored.
pub fn now_unix_seconds() -> u64 {
    now_unix_nanos() / 1_000_000_000
}

/// Returns raw monotonic nanoseconds since boot.
///
/// Prefer using `now()` which returns a type-safe `Instant`.
#[inline]
pub fn monotonic_ns() -> u64 {
    pal::clock::monotonic_ns()
}

/// Sleep for the specified duration.
///
/// Accepts both `abi::types::instant::Duration` and `core::time::Duration`.
pub fn sleep(duration: impl Into<Duration>) {
    pal::clock::sleep_ns(duration.into().as_nanos());
}

/// Sleep for the specified number of milliseconds.
pub fn sleep_ms(ms: u64) {
    pal::clock::sleep_ns(ms * 1_000_000);
}

/// Sleep for the specified number of nanoseconds.
#[inline]
pub fn sleep_ns(ns: u64) {
    pal::clock::sleep_ns(ns);
}
