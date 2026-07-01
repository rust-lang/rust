use core::num::niche_types::Nanoseconds;

use crate::cmp::Ordering;
use crate::hash::{Hash, Hasher};
use crate::sys::pal::time::{
    INTERVALS_PER_SEC, checked_dur2intervals, intervals2dur, perf_counter,
};
use crate::sys::{IntoInner, c};
use crate::time::{Duration, Instant};
use crate::{fmt, mem};

const NANOS_PER_SEC: u64 = 1_000_000_000;

pub fn now() -> Instant {
    // High precision timing on windows operates in "Performance Counter"
    // units, as returned by the WINAPI QueryPerformanceCounter function.
    // These relate to seconds by a factor of QueryPerformanceFrequency.

    let freq = perf_counter::frequency();
    let counts = perf_counter::now();

    let secs = counts.div_euclid(freq);
    let subsec_counts = counts.rem_euclid(freq) as u64;

    // "QPC does not go backward.", so the frequency cannot be negative.
    let freq = freq as u64;
    let nanos = Nanoseconds::new((NANOS_PER_SEC.strict_mul(subsec_counts) / freq) as u32).unwrap();

    Instant { secs, nanos }
}

pub use perf_counter::epsilon;

#[derive(Copy, Clone)]
pub struct SystemTime {
    t: c::FILETIME,
}

pub const UNIX_EPOCH: SystemTime =
    SystemTime::from_intervals(11_644_473_600 * INTERVALS_PER_SEC as i64);

impl SystemTime {
    pub const MAX: SystemTime = SystemTime::from_intervals(i64::MAX);
    pub const MIN: SystemTime = SystemTime::from_intervals(0);

    pub fn now() -> SystemTime {
        unsafe {
            let mut t: SystemTime = mem::zeroed();
            c::GetSystemTimePreciseAsFileTime(&mut t.t);
            t
        }
    }

    const fn from_intervals(intervals: i64) -> SystemTime {
        SystemTime {
            t: c::FILETIME {
                dwLowDateTime: intervals as u32,
                dwHighDateTime: (intervals >> 32) as u32,
            },
        }
    }

    fn intervals(&self) -> i64 {
        (self.t.dwLowDateTime as i64) | ((self.t.dwHighDateTime as i64) << 32)
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        let me = self.intervals();
        let other = other.intervals();
        if me >= other {
            Ok(intervals2dur((me - other) as u64))
        } else {
            Err(intervals2dur((other - me) as u64))
        }
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        let intervals = self.intervals().checked_add(checked_dur2intervals(other)?)?;
        Some(SystemTime::from_intervals(intervals))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        // Windows does not support times before 1601, hence why we don't
        // support negatives. In order to tackle this, we try to convert the
        // resulting value into an u64, which should obviously fail in the case
        // that the value is below zero.
        let intervals: u64 =
            self.intervals().checked_sub(checked_dur2intervals(other)?)?.try_into().ok()?;
        Some(SystemTime::from_intervals(intervals as i64))
    }
}

impl PartialEq for SystemTime {
    fn eq(&self, other: &SystemTime) -> bool {
        self.intervals() == other.intervals()
    }
}

impl Eq for SystemTime {}

impl PartialOrd for SystemTime {
    fn partial_cmp(&self, other: &SystemTime) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SystemTime {
    fn cmp(&self, other: &SystemTime) -> Ordering {
        self.intervals().cmp(&other.intervals())
    }
}

impl fmt::Debug for SystemTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemTime").field("intervals", &self.intervals()).finish()
    }
}

impl From<c::FILETIME> for SystemTime {
    fn from(t: c::FILETIME) -> SystemTime {
        SystemTime { t }
    }
}

impl IntoInner<c::FILETIME> for SystemTime {
    fn into_inner(self) -> c::FILETIME {
        self.t
    }
}

impl Hash for SystemTime {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.intervals().hash(state)
    }
}
