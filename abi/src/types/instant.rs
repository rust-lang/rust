//! Monotonic time primitives.
//!
//! Provides type-safe `Instant` and `Duration` types with guaranteed monotonic
//! behavior. All time in Thing-OS derives from this single timebase.

use crate::{BlobId, SymbolId, ThingId};
/// Monotonic timestamp in nanoseconds since system boot.
///
/// # Guarantees
/// - Always increasing (never goes backward)
/// - Never negative
/// - Consistent across kernel and userspace
///
/// # Wire Format
/// `#[repr(transparent)]` ensures this is ABI-compatible with a raw `u64`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Instant(pub u64);

/// Time duration in nanoseconds.
///
/// Represents the difference between two `Instant` values or a delay period.
///
/// # Wire Format
/// `#[repr(transparent)]` ensures this is ABI-compatible with a raw `u64`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Duration(pub u64);

// ============================================================================
// Instant Implementation
// ============================================================================

impl Instant {
    /// The epoch instant (time zero at boot).
    pub const ZERO: Instant = Instant(0);

    /// Create an instant from raw nanoseconds.
    #[inline]
    pub const fn from_nanos(ns: u64) -> Self {
        Self(ns)
    }

    /// Returns the total nanoseconds since boot.
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Returns the total milliseconds since boot (truncated).
    #[inline]
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }

    /// Returns the total seconds since boot (truncated).
    #[inline]
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Returns the duration elapsed since an earlier instant.
    ///
    /// Returns `None` if `earlier` is after `self` (which should not happen
    /// with monotonic time, but provides safety).
    #[inline]
    pub const fn checked_sub(self, earlier: Instant) -> Option<Duration> {
        match self.0.checked_sub(earlier.0) {
            Some(ns) => Some(Duration(ns)),
            None => None,
        }
    }

    /// Returns the duration elapsed since an earlier instant, saturating at zero.
    #[inline]
    pub const fn saturating_sub(self, earlier: Instant) -> Duration {
        Duration(self.0.saturating_sub(earlier.0))
    }

    /// Returns this instant plus a duration.
    ///
    /// Returns `None` on overflow.
    #[inline]
    pub const fn checked_add(self, duration: Duration) -> Option<Instant> {
        match self.0.checked_add(duration.0) {
            Some(ns) => Some(Instant(ns)),
            None => None,
        }
    }

    /// Returns this instant plus a duration, saturating at `u64::MAX`.
    #[inline]
    pub const fn saturating_add(self, duration: Duration) -> Instant {
        Instant(self.0.saturating_add(duration.0))
    }

    /// Returns the duration between two instants (absolute difference).
    #[inline]
    pub const fn abs_diff(self, other: Instant) -> Duration {
        if self.0 >= other.0 {
            Duration(self.0 - other.0)
        } else {
            Duration(other.0 - self.0)
        }
    }
}

// ============================================================================
// Duration Implementation
// ============================================================================

impl Duration {
    /// Zero duration.
    pub const ZERO: Duration = Duration(0);

    /// One millisecond.
    pub const MILLISECOND: Duration = Duration(1_000_000);

    /// One second.
    pub const SECOND: Duration = Duration(1_000_000_000);

    /// Create a duration from nanoseconds.
    #[inline]
    pub const fn from_nanos(ns: u64) -> Self {
        Self(ns)
    }

    /// Create a duration from microseconds.
    #[inline]
    pub const fn from_micros(us: u64) -> Self {
        Self(us.saturating_mul(1_000))
    }

    /// Create a duration from milliseconds.
    #[inline]
    pub const fn from_millis(ms: u64) -> Self {
        Self(ms.saturating_mul(1_000_000))
    }

    /// Create a duration from seconds.
    #[inline]
    pub const fn from_secs(s: u64) -> Self {
        Self(s.saturating_mul(1_000_000_000))
    }

    /// Returns the total nanoseconds.
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Returns the total microseconds (truncated).
    #[inline]
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }

    /// Returns the total milliseconds (truncated).
    #[inline]
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }

    /// Returns the total seconds (truncated).
    #[inline]
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Returns the nanoseconds component (0-999_999_999).
    #[inline]
    pub const fn subsec_nanos(&self) -> u32 {
        (self.0 % 1_000_000_000) as u32
    }

    /// Returns `true` if this duration is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Checked addition.
    #[inline]
    pub const fn checked_add(self, rhs: Duration) -> Option<Duration> {
        match self.0.checked_add(rhs.0) {
            Some(ns) => Some(Duration(ns)),
            None => None,
        }
    }

    /// Saturating addition.
    #[inline]
    pub const fn saturating_add(self, rhs: Duration) -> Duration {
        Duration(self.0.saturating_add(rhs.0))
    }

    /// Checked subtraction.
    #[inline]
    pub const fn checked_sub(self, rhs: Duration) -> Option<Duration> {
        match self.0.checked_sub(rhs.0) {
            Some(ns) => Some(Duration(ns)),
            None => None,
        }
    }

    /// Saturating subtraction.
    #[inline]
    pub const fn saturating_sub(self, rhs: Duration) -> Duration {
        Duration(self.0.saturating_sub(rhs.0))
    }

    /// Checked multiplication.
    #[inline]
    pub const fn checked_mul(self, rhs: u64) -> Option<Duration> {
        match self.0.checked_mul(rhs) {
            Some(ns) => Some(Duration(ns)),
            None => None,
        }
    }

    /// Saturating multiplication.
    #[inline]
    pub const fn saturating_mul(self, rhs: u64) -> Duration {
        Duration(self.0.saturating_mul(rhs))
    }

    /// Checked division.
    #[inline]
    pub const fn checked_div(self, rhs: u64) -> Option<Duration> {
        match self.0.checked_div(rhs) {
            Some(ns) => Some(Duration(ns)),
            None => None,
        }
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl core::ops::Add<Duration> for Instant {
    type Output = Instant;

    #[inline]
    fn add(self, rhs: Duration) -> Instant {
        self.saturating_add(rhs)
    }
}

impl core::ops::Sub<Instant> for Instant {
    type Output = Duration;

    #[inline]
    fn sub(self, rhs: Instant) -> Duration {
        self.saturating_sub(rhs)
    }
}

impl core::ops::Add<Duration> for Duration {
    type Output = Duration;

    #[inline]
    fn add(self, rhs: Duration) -> Duration {
        self.saturating_add(rhs)
    }
}

impl core::ops::Sub<Duration> for Duration {
    type Output = Duration;

    #[inline]
    fn sub(self, rhs: Duration) -> Duration {
        self.saturating_sub(rhs)
    }
}

// ============================================================================
// Conversions
// ============================================================================

impl From<u64> for Instant {
    #[inline]
    fn from(ns: u64) -> Self {
        Instant(ns)
    }
}

impl From<Instant> for u64 {
    #[inline]
    fn from(i: Instant) -> u64 {
        i.0
    }
}

impl From<u64> for Duration {
    #[inline]
    fn from(ns: u64) -> Self {
        Duration(ns)
    }
}

impl From<Duration> for u64 {
    #[inline]
    fn from(d: Duration) -> u64 {
        d.0
    }
}

// Conversion from core::time::Duration (available in no_std)
impl From<core::time::Duration> for Duration {
    #[inline]
    fn from(d: core::time::Duration) -> Self {
        Duration(d.as_nanos() as u64)
    }
}

impl From<Duration> for core::time::Duration {
    #[inline]
    fn from(d: Duration) -> Self {
        core::time::Duration::from_nanos(d.0)
    }
}
