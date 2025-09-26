use crate::cmp::Ordering;
use crate::hash::{Hash, Hasher};
use crate::num::NonZero;
use crate::ops::Neg;
use crate::ptr::null;
use crate::sync::atomic::Ordering::Relaxed;
use crate::sync::atomic::{Atomic, AtomicU64};
use crate::sys::pal::{c, cvt};
use crate::sys_common::IntoInner;
use crate::time::Duration;
use crate::{fmt, mem};

const NANOS_PER_SEC: u64 = 1_000_000_000;
const INTERVALS_PER_SEC: u64 = NANOS_PER_SEC / 100;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Instant {
    // High precision timing on windows operates in "Performance Counter"
    // units, as returned by the WINAPI QueryPerformanceCounter function.
    // These relate to seconds by a factor of QueryPerformanceFrequency.
    ticks: i64,
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
        let mut ticks: i64 = 0;
        cvt(unsafe { c::QueryPerformanceCounter(&mut ticks) }).unwrap();
        Instant { ticks }
    }

    fn frequency() -> NonZero<u64> {
        // Either the cached result of `QueryPerformanceFrequency` or `0` for
        // uninitialized. Storing this as a single `AtomicU64` allows us to use
        // `Relaxed` operations, as we are only interested in the effects on a
        // single memory location.
        static FREQUENCY: Atomic<u64> = AtomicU64::new(0);

        let cached = FREQUENCY.load(Relaxed);
        // If a previous thread has filled in this global state, use that.
        if let Some(cached) = NonZero::new(cached) {
            return cached;
        }

        // ... otherwise learn for ourselves ...
        let mut frequency = 0;
        unsafe {
            cvt(c::QueryPerformanceFrequency(&mut frequency)).unwrap();
        }

        let frequency = NonZero::new(frequency.cast_unsigned())
            .expect("frequency of performance counter should not be zero");

        // Check that the frequency can be safely multiplied with `NANOS_PER_SECOND`.
        // This should always succeed since the performance counter is documented not
        // to wrap around for 100 years, so the maximum frequency is less than 6 billion
        // ticks per second, which in turn means that this multiplication will not overflow.
        frequency
            .get()
            .checked_mul(NANOS_PER_SEC)
            .expect("wraparound must only occur after 100 years");

        FREQUENCY.store(frequency.get(), Relaxed);
        frequency
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        if self.ticks >= other.ticks {
            // Find the difference by performing an unsigned subtraction, as
            // the subtraction might overflow otherwise.
            let ticks = self.ticks.cast_unsigned().wrapping_sub(other.ticks.cast_unsigned());
            let freq = Self::frequency();
            let secs = ticks / freq;
            let subsec_ticks = ticks % freq;
            // SAFETY:
            // `subsec_ticks` is smaller than `freq`, and `freq` can be multiplied
            // with `NANOS_PER_SEC` without overflow (we checked that in the
            // `frequency` function above).
            let subsec_nanos = unsafe { NANOS_PER_SEC.unchecked_mul(subsec_ticks) / freq };
            Some(Duration::new(secs, subsec_nanos as u32))
        } else if other.ticks == self.ticks + 1 {
            // Per microsoft docs, the margin of error for cross-thread time
            // comparisons using QueryPerformanceCounter is one tick, hence we
            // consider `other` to be equal to `self` if the difference is only
            // one tick.
            // Reference: https://docs.microsoft.com/en-us/windows/desktop/SysInfo/acquiring-high-resolution-time-stamps
            Some(Duration::ZERO)
        } else {
            None
        }
    }

    fn duration2ticks(dur: Duration) -> Option<u64> {
        let freq = Self::frequency();
        let whole = dur.as_secs().checked_mul(freq.get())?;
        // SAFETY:
        // `subsec_nanos` is smaller than `NANOS_PER_SEC` and `freq` can be
        // multiplied with `NANOS_PER_SEC` without overflow (we checked that in the
        // `frequency` function above).
        let frac = unsafe { freq.get().unchecked_mul(dur.subsec_nanos().into()) } / NANOS_PER_SEC;
        whole.checked_add(frac)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        let to_add = Self::duration2ticks(*other)?;
        let ticks = self.ticks.checked_add_unsigned(to_add)?;
        Some(Instant { ticks })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        let to_sub = Self::duration2ticks(*other)?;
        let ticks = self.ticks.checked_sub_unsigned(to_sub)?;
        Some(Instant { ticks })
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

/// A timer you can wait on.
pub(crate) struct WaitableTimer {
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
