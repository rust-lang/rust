use super::{abi, error::expect_success};
use crate::{convert::TryInto, mem::MaybeUninit, time::Duration};

pub use super::itron::time::Instant;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(abi::time_t);

pub const UNIX_EPOCH: SystemTime = SystemTime(0);

impl SystemTime {
    pub fn now() -> SystemTime {
        let rtc = unsafe {
            let mut out = MaybeUninit::zeroed();
            expect_success(abi::SOLID_RTC_ReadTime(out.as_mut_ptr()), &"SOLID_RTC_ReadTime");
            out.assume_init()
        };
        let t = unsafe {
            libc::mktime(&mut libc::tm {
                tm_sec: rtc.tm_sec,
                tm_min: rtc.tm_min,
                tm_hour: rtc.tm_hour,
                tm_mday: rtc.tm_mday,
                tm_mon: rtc.tm_mon,
                tm_year: rtc.tm_year,
                tm_wday: rtc.tm_wday,
                tm_yday: 0,
                tm_isdst: 0,
                tm_gmtoff: 0,
                tm_zone: crate::ptr::null_mut(),
            })
        };
        assert_ne!(t, -1, "mktime failed");
        SystemTime(t)
    }

    pub(super) fn from_time_t(t: abi::time_t) -> Self {
        Self(t)
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        if self.0 >= other.0 {
            Ok(Duration::from_secs((self.0 as u64).wrapping_sub(other.0 as u64)))
        } else {
            Err(Duration::from_secs((other.0 as u64).wrapping_sub(self.0 as u64)))
        }
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add(other.as_secs().try_into().ok()?)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub(other.as_secs().try_into().ok()?)?))
    }
}
