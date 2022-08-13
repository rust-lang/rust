//! Time is implemeted using `EFI_RUNTIME_SERVICES.GetTime()`
//! While this is  not technically monotonic, the single-threaded nature of UEFI might make it fine
//! to use for Instant. Still, maybe revisit this in future.

use crate::os::uefi;
use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::ZERO);

impl Instant {
    pub fn now() -> Instant {
        if let Some(runtime_services) = uefi::env::get_runtime_services() {
            let mut t = r_efi::efi::Time::default();
            let r =
                unsafe { ((*runtime_services.as_ptr()).get_time)(&mut t, crate::ptr::null_mut()) };

            if r.is_error() {
                panic!("time not implemented on this platform")
            } else {
                Instant(uefi_time_to_duration(t))
            }
        } else {
            panic!("Runtime Services are needed for Time to work")
        }
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

// Using Unix representation of Time.
impl SystemTime {
    pub fn now() -> SystemTime {
        if let Some(runtime_services) = uefi::env::get_runtime_services() {
            let mut t = r_efi::efi::Time::default();
            let r =
                unsafe { ((*runtime_services.as_ptr()).get_time)(&mut t, crate::ptr::null_mut()) };

            if r.is_error() {
                panic!("time not implemented on this platform")
            } else {
                SystemTime::from(t)
            }
        } else {
            panic!("Runtime Services are needed for Time to work")
        }
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

impl From<r_efi::system::Time> for SystemTime {
    fn from(t: r_efi::system::Time) -> Self {
        SystemTime(uefi_time_to_duration(t))
    }
}

// FIXME: Don't know how to use Daylight Saving thing
fn uefi_time_to_duration(t: r_efi::system::Time) -> Duration {
    const SEC_IN_MIN: u64 = 60;
    const SEC_IN_HOUR: u64 = SEC_IN_MIN * 60;
    const SEC_IN_DAY: u64 = SEC_IN_HOUR * 24;
    const SEC_IN_YEAR: u64 = SEC_IN_DAY * 365;
    const MONTH_DAYS: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 211, 242, 272, 303, 333];

    let localtime_epoch: u64 = u64::from(t.year - 1970) * SEC_IN_YEAR
        + u64::from((t.year - 1968) / 4) * SEC_IN_DAY
        + MONTH_DAYS[usize::from(t.month - 1)] * SEC_IN_DAY
        + u64::from(t.day - 1) * SEC_IN_DAY
        + u64::from(t.hour) * SEC_IN_HOUR
        + u64::from(t.minute) * SEC_IN_MIN
        + u64::from(t.second);
    let timezone_epoch: i64 = i64::from(t.timezone) * (SEC_IN_MIN as i64);
    let utc_epoch: u64 = ((localtime_epoch as i64) + timezone_epoch) as u64;

    Duration::new(utc_epoch, t.nanosecond)
}
