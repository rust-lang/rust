use core::hash::{Hash, Hasher};
use core::ops::Neg;

use crate::cmp::Ordering;
use crate::ptr::null;
use crate::sys::c;
use crate::sys_common::IntoInner;
use crate::time::Duration;
use crate::{fmt, mem};

const NANOS_PER_SEC: u64 = 1_000_000_000;
const INTERVALS_PER_SEC: u64 = NANOS_PER_SEC / 100;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Instant {
    // This duration is relative to an arbitrary microsecond epoch
    // from the winapi QueryPerformanceCounter function.
    t: Duration,
}

#[derive(Copy, Clone)]
pub struct SystemTime {
    t: c::FILETIME,
}

const INTERVALS_TO_UNIX_EPOCH: u64 = 11_644_473_600 * INTERVALS_PER_SEC;

pub const UNIX_EPOCH: SystemTime = SystemTime {
    t: c::FILETIME {
        dwLowDateTime: INTERVALS_TO_UNIX_EPOCH as u32,
        dwHighDateTime: (INTERVALS_TO_UNIX_EPOCH >> 32) as u32,
    },
};

impl Instant {
    pub fn now() -> Instant {
        // High precision timing on windows operates in "Performance Counter"
        // units, as returned by the WINAPI QueryPerformanceCounter function.
        // These relate to seconds by a factor of QueryPerformanceFrequency.
        // In order to keep unit conversions out of normal interval math, we
        // measure in QPC units and immediately convert to nanoseconds.
        perf_counter::PerformanceCounterInstant::now().into()
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        // On windows there's a threshold below which we consider two timestamps
        // equivalent due to measurement error. For more details + doc link,
        // check the docs on epsilon.
        let epsilon = perf_counter::PerformanceCounterInstant::epsilon();
        if other.t > self.t && other.t - self.t <= epsilon {
            Some(Duration::new(0, 0))
        } else {
            self.t.checked_sub(other.t)
        }
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant { t: self.t.checked_add(*other)? })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant { t: self.t.checked_sub(*other)? })
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        unsafe {
            let mut t: SystemTime = mem::zeroed();
            c::GetSystemTimePreciseAsFileTime(&mut t.t);
            t
        }
    }

    fn from_intervals(intervals: i64) -> SystemTime {
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
        let intervals = self.intervals().checked_sub(checked_dur2intervals(other)?)?;
        Some(SystemTime::from_intervals(intervals))
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

fn checked_dur2intervals(dur: &Duration) -> Option<i64> {
    dur.as_secs()
        .checked_mul(INTERVALS_PER_SEC)?
        .checked_add(dur.subsec_nanos() as u64 / 100)?
        .try_into()
        .ok()
}

fn intervals2dur(intervals: u64) -> Duration {
    Duration::new(intervals / INTERVALS_PER_SEC, ((intervals % INTERVALS_PER_SEC) * 100) as u32)
}

mod perf_counter {
    use super::NANOS_PER_SEC;
    use crate::sync::atomic::{Atomic, AtomicU64, Ordering};
    use crate::sys::{c, cvt};
    use crate::sys_common::mul_div_u64;
    use crate::time::Duration;

    pub struct PerformanceCounterInstant {
        ts: i64,
    }
    impl PerformanceCounterInstant {
        pub fn now() -> Self {
            Self { ts: query() }
        }

        // Per microsoft docs, the margin of error for cross-thread time comparisons
        // using QueryPerformanceCounter is 1 "tick" -- defined as 1/frequency().
        // Reference: https://docs.microsoft.com/en-us/windows/desktop/SysInfo
        //                   /acquiring-high-resolution-time-stamps
        pub fn epsilon() -> Duration {
            let epsilon = NANOS_PER_SEC / (frequency() as u64);
            Duration::from_nanos(epsilon)
        }
    }
    impl From<PerformanceCounterInstant> for super::Instant {
        fn from(other: PerformanceCounterInstant) -> Self {
            let freq = frequency() as u64;
            let instant_nsec = mul_div_u64(other.ts as u64, NANOS_PER_SEC, freq);
            Self { t: Duration::from_nanos(instant_nsec) }
        }
    }

    fn frequency() -> i64 {
        // Either the cached result of `QueryPerformanceFrequency` or `0` for
        // uninitialized. Storing this as a single `AtomicU64` allows us to use
        // `Relaxed` operations, as we are only interested in the effects on a
        // single memory location.
        static FREQUENCY: Atomic<u64> = AtomicU64::new(0);

        let cached = FREQUENCY.load(Ordering::Relaxed);
        // If a previous thread has filled in this global state, use that.
        if cached != 0 {
            return cached as i64;
        }
        // ... otherwise learn for ourselves ...
        let mut frequency = 0;
        unsafe {
            cvt(c::QueryPerformanceFrequency(&mut frequency)).unwrap();
        }

        FREQUENCY.store(frequency as u64, Ordering::Relaxed);
        frequency
    }

    fn query() -> i64 {
        let mut qpc_value: i64 = 0;
        cvt(unsafe { c::QueryPerformanceCounter(&mut qpc_value) }).unwrap();
        qpc_value
    }
}

/// A timer you can wait on.
pub(super) struct WaitableTimer {
    handle: c::HANDLE,
}
impl WaitableTimer {
    /// Creates a high-resolution timer. Will fail before Windows 10, version 1803.
    pub fn high_resolution() -> Result<Self, ()> {
        let handle = unsafe {
            c::CreateWaitableTimerExW(
                null(),
                null(),
                c::CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                c::TIMER_ALL_ACCESS,
            )
        };
        if !handle.is_null() { Ok(Self { handle }) } else { Err(()) }
    }
    pub fn set(&self, duration: Duration) -> Result<(), ()> {
        // Convert the Duration to a format similar to FILETIME.
        // Negative values are relative times whereas positive values are absolute.
        // Therefore we negate the relative duration.
        let time = checked_dur2intervals(&duration).ok_or(())?.neg();
        let result = unsafe { c::SetWaitableTimer(self.handle, &time, 0, None, null(), c::FALSE) };
        if result != 0 { Ok(()) } else { Err(()) }
    }
    pub fn wait(&self) -> Result<(), ()> {
        let result = unsafe { c::WaitForSingleObject(self.handle, c::INFINITE) };
        if result != c::WAIT_FAILED { Ok(()) } else { Err(()) }
    }
}
impl Drop for WaitableTimer {
    fn drop(&mut self) {
        unsafe { c::CloseHandle(self.handle) };
    }
}
