use cmp::Ordering;
use fmt;
use mem;
use sync::Once;
use sys::c;
use sys::cvt;
use sys_common::mul_div_u64;
use time::Duration;
use convert::TryInto;
use core::hash::{Hash, Hasher};

const NANOS_PER_SEC: u64 = 1_000_000_000;
const INTERVALS_PER_SEC: u64 = NANOS_PER_SEC / 100;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Instant {
    t: c::LARGE_INTEGER,
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
        let mut t = Instant { t: 0 };
        cvt(unsafe {
            c::QueryPerformanceCounter(&mut t.t)
        }).unwrap();
        t
    }

    pub fn sub_instant(&self, other: &Instant) -> Duration {
        // Values which are +- 1 need to be considered as basically the same
        // units in time due to various measurement oddities, according to
        // Windows [1]
        //
        // [1]:
        // https://msdn.microsoft.com/en-us/library/windows/desktop
        //                           /dn553408%28v=vs.85%29.aspx#guidance
        if other.t > self.t && other.t - self.t == 1 {
            return Duration::new(0, 0)
        }
        let diff = (self.t as u64).checked_sub(other.t as u64)
                                  .expect("specified instant was later than \
                                           self");
        let nanos = mul_div_u64(diff, NANOS_PER_SEC, frequency() as u64);
        Duration::new(nanos / NANOS_PER_SEC, (nanos % NANOS_PER_SEC) as u32)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        let freq = frequency() as u64;
        let t = other.as_secs()
            .checked_mul(freq)?
            .checked_add(mul_div_u64(other.subsec_nanos() as u64, freq, NANOS_PER_SEC))?
            .checked_add(self.t as u64)?;
        Some(Instant {
            t: t as c::LARGE_INTEGER,
        })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        let freq = frequency() as u64;
        let t = other.as_secs().checked_mul(freq).and_then(|i| {
            (self.t as u64).checked_sub(i)
        }).and_then(|i| {
            i.checked_sub(mul_div_u64(other.subsec_nanos() as u64, freq, NANOS_PER_SEC))
        })?;
        Some(Instant {
            t: t as c::LARGE_INTEGER,
        })
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        unsafe {
            let mut t: SystemTime = mem::zeroed();
            c::GetSystemTimeAsFileTime(&mut t.t);
            return t
        }
    }

    fn from_intervals(intervals: i64) -> SystemTime {
        SystemTime {
            t: c::FILETIME {
                dwLowDateTime: intervals as c::DWORD,
                dwHighDateTime: (intervals >> 32) as c::DWORD,
            }
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SystemTime")
         .field("intervals", &self.intervals())
         .finish()
    }
}

impl From<c::FILETIME> for SystemTime {
    fn from(t: c::FILETIME) -> SystemTime {
        SystemTime { t }
    }
}

impl Hash for SystemTime {
    fn hash<H : Hasher>(&self, state: &mut H) {
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
    Duration::new(intervals / INTERVALS_PER_SEC,
                  ((intervals % INTERVALS_PER_SEC) * 100) as u32)
}

fn frequency() -> c::LARGE_INTEGER {
    static mut FREQUENCY: c::LARGE_INTEGER = 0;
    static ONCE: Once = Once::new();

    unsafe {
        ONCE.call_once(|| {
            cvt(c::QueryPerformanceFrequency(&mut FREQUENCY)).unwrap();
        });
        FREQUENCY
    }
}
