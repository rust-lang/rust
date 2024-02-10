use crate::time::Duration;

const SECS_IN_MINUTE: u64 = 60;
const SECS_IN_HOUR: u64 = SECS_IN_MINUTE * 60;
const SECS_IN_DAY: u64 = SECS_IN_HOUR * 24;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::from_secs(0));

impl Instant {
    pub fn now() -> Instant {
        panic!("time not implemented on this platform")
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
}

impl SystemTime {
    pub fn now() -> SystemTime {
        system_time_internal::now()
            .unwrap_or_else(|| panic!("time not implemented on this platform"))
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

pub(crate) mod system_time_internal {
    use super::super::helpers;
    use super::*;
    use crate::mem::MaybeUninit;
    use crate::ptr::NonNull;
    use r_efi::efi::{RuntimeServices, Time};

    pub fn now() -> Option<SystemTime> {
        let runtime_services: NonNull<RuntimeServices> = helpers::runtime_services()?;
        let mut t: MaybeUninit<Time> = MaybeUninit::uninit();
        let r = unsafe {
            ((*runtime_services.as_ptr()).get_time)(t.as_mut_ptr(), crate::ptr::null_mut())
        };

        if r.is_error() {
            return None;
        }

        let t = unsafe { t.assume_init() };

        Some(SystemTime(uefi_time_to_duration(t)))
    }

    // This algorithm is based on the one described in the post
    // https://blog.reverberate.org/2020/05/12/optimizing-date-algorithms.html
    pub const fn uefi_time_to_duration(t: r_efi::system::Time) -> Duration {
        assert!(t.month <= 12);
        assert!(t.month != 0);

        const YEAR_BASE: u32 = 4800; /* Before min year, multiple of 400. */

        // Calculate the number of days since 1/1/1970
        // Use 1 March as the start
        let (m_adj, overflow): (u32, bool) = (t.month as u32).overflowing_sub(3);
        let (carry, adjust): (u32, u32) = if overflow { (1, 12) } else { (0, 0) };
        let y_adj: u32 = (t.year as u32) + YEAR_BASE - carry;
        let month_days: u32 = (m_adj.wrapping_add(adjust) * 62719 + 769) / 2048;
        let leap_days: u32 = y_adj / 4 - y_adj / 100 + y_adj / 400;
        let days: u32 = y_adj * 365 + leap_days + month_days + (t.day as u32 - 1) - 2472632;

        let localtime_epoch: u64 = (days as u64) * SECS_IN_DAY
            + (t.second as u64)
            + (t.minute as u64) * SECS_IN_MINUTE
            + (t.hour as u64) * SECS_IN_HOUR;

        let utc_epoch: u64 = if t.timezone == r_efi::efi::UNSPECIFIED_TIMEZONE {
            localtime_epoch
        } else {
            (localtime_epoch as i64 + (t.timezone as i64) * SECS_IN_MINUTE as i64) as u64
        };

        Duration::new(utc_epoch, t.nanosecond)
    }
}
