use crate::io;
use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Instant {
        Instant(Duration::from_nanos(wasip2::clocks::monotonic_clock::now()))
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub(*other)?))
    }

    pub(crate) fn as_duration(&self) -> &Duration {
        &self.0
    }
}

impl SystemTime {
    pub fn now() -> SystemTime {
        let now = wasip2::clocks::wall_clock::now();
        SystemTime(Duration::new(now.seconds, now.nanoseconds))
    }

    pub fn from_timespec(ts: libc::timespec) -> SystemTime {
        SystemTime(Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32))
    }

    pub fn to_timespec(&self) -> io::Result<libc::timespec> {
        Ok(libc::timespec {
            tv_sec: self
                .0
                .as_secs()
                .try_into()
                .map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW))?,
            tv_nsec: self
                .0
                .subsec_nanos()
                .try_into()
                .map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW))?,
        })
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub(*other)?))
    }
}
