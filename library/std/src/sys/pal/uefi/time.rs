use crate::time::Duration;

const SECS_IN_MINUTE: u64 = 60;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

/// When a Timezone is specified, the stored Duration is in UTC. If timezone is unspecified, then
/// the timezone is assumed to be in UTC.
///
/// UEFI SystemTime is stored as Duration from 1900-01-01-00:00:00 with timezone -1440 as anchor
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime::from_uefi(r_efi::efi::Time {
    year: 1970,
    month: 1,
    day: 1,
    hour: 0,
    minute: 0,
    second: 0,
    nanosecond: 0,
    timezone: 0,
    daylight: 0,
    pad1: 0,
    pad2: 0,
})
.unwrap();

const MAX_UEFI_TIME: SystemTime = SystemTime::from_uefi(r_efi::efi::Time {
    year: 9999,
    month: 12,
    day: 31,
    hour: 23,
    minute: 59,
    second: 59,
    nanosecond: 999_999_999,
    timezone: 1440,
    daylight: 0,
    pad1: 0,
    pad2: 0,
})
.unwrap();

impl Instant {
    pub fn now() -> Instant {
        // If we have a timestamp protocol, use it.
        if let Some(x) = instant_internal::timestamp_protocol() {
            return x;
        }

        if let Some(x) = instant_internal::platform_specific() {
            return x;
        }

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
    pub const MAX: SystemTime = MAX_UEFI_TIME;

    pub const MIN: SystemTime = SystemTime::from_uefi(r_efi::efi::Time {
        year: 1900,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        nanosecond: 0,
        timezone: -1440,
        daylight: 0,
        pad1: 0,
        pad2: 0,
    })
    .unwrap();

    pub(crate) const fn from_uefi(t: r_efi::efi::Time) -> Option<Self> {
        match system_time_internal::from_uefi(&t) {
            Some(x) => Some(Self(x)),
            None => None,
        }
    }

    pub(crate) const fn to_uefi(
        self,
        timezone: i16,
        daylight: u8,
    ) -> Result<r_efi::efi::Time, i16> {
        // system_time_internal::to_uefi requires a valid timezone. In case of unspecified timezone,
        // we just pass 0 since it is assumed that no timezone related adjustments are required.
        if timezone == r_efi::efi::UNSPECIFIED_TIMEZONE {
            system_time_internal::to_uefi(&self.0, 0, daylight)
        } else {
            system_time_internal::to_uefi(&self.0, timezone, daylight)
        }
    }

    /// Create UEFI Time with the closest timezone (minute offset) that still allows the time to be
    /// represented.
    pub(crate) fn to_uefi_loose(self, timezone: i16, daylight: u8) -> r_efi::efi::Time {
        match self.to_uefi(timezone, daylight) {
            Ok(x) => x,
            Err(tz) => self.to_uefi(tz, daylight).unwrap(),
        }
    }

    pub fn now() -> SystemTime {
        Self::from_uefi(system_time_internal::now())
            .expect("time incorrectly implemented on this platform")
    }

    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        let temp = Self(self.0.checked_add(*other)?);

        // Check if can be represented in UEFI
        if temp <= MAX_UEFI_TIME { Some(temp) } else { None }
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        self.0.checked_sub(*other).map(Self)
    }
}

pub(crate) mod system_time_internal {
    use r_efi::efi::{RuntimeServices, Time};

    use super::super::helpers;
    use super::*;
    use crate::mem::MaybeUninit;
    use crate::ptr::NonNull;

    const SECS_IN_HOUR: u64 = SECS_IN_MINUTE * 60;
    const SECS_IN_DAY: u64 = SECS_IN_HOUR * 24;
    const SYSTEMTIME_TIMEZONE: i64 = -1440 * SECS_IN_MINUTE as i64;

    pub(crate) fn now() -> Time {
        let runtime_services: NonNull<RuntimeServices> =
            helpers::runtime_services().expect("Runtime services are not available");
        let mut t: MaybeUninit<Time> = MaybeUninit::uninit();
        let r = unsafe {
            ((*runtime_services.as_ptr()).get_time)(t.as_mut_ptr(), crate::ptr::null_mut())
        };
        if r.is_error() {
            panic!("time not implemented on this platform");
        }

        unsafe { t.assume_init() }
    }

    /// This algorithm is a modified form of the one described in the post
    /// https://blog.reverberate.org/2020/05/12/optimizing-date-algorithms.html
    ///
    /// The changes are to use 1900-01-01-00:00:00 with timezone -1440 as anchor instead of UNIX
    /// epoch used in the original algorithm.
    pub(crate) const fn from_uefi(t: &Time) -> Option<Duration> {
        if !(t.month <= 12
            && t.month != 0
            && t.year >= 1900
            && t.year <= 9999
            && t.day <= 31
            && t.day != 0
            && t.second < 60
            && t.minute <= 60
            && t.hour < 24
            && t.nanosecond < 1_000_000_000
            && ((t.timezone <= 1440 && t.timezone >= -1440)
                || t.timezone == r_efi::efi::UNSPECIFIED_TIMEZONE))
        {
            return None;
        }

        const YEAR_BASE: u32 = 4800; /* Before min year, multiple of 400. */

        // Calculate the number of days since 1/1/1900. This is the earliest supported date in UEFI
        // time.
        // Use 1 March as the start
        let (m_adj, overflow): (u32, bool) = (t.month as u32).overflowing_sub(3);
        let (carry, adjust): (u32, u32) = if overflow { (1, 12) } else { (0, 0) };
        let y_adj: u32 = (t.year as u32) + YEAR_BASE - carry;
        let month_days: u32 = (m_adj.wrapping_add(adjust) * 62719 + 769) / 2048;
        let leap_days: u32 = y_adj / 4 - y_adj / 100 + y_adj / 400;
        let days: u32 = y_adj * 365 + leap_days + month_days + (t.day as u32 - 1) - 2447065;

        let localtime_epoch: u64 = (days as u64) * SECS_IN_DAY
            + (t.second as u64)
            + (t.minute as u64) * SECS_IN_MINUTE
            + (t.hour as u64) * SECS_IN_HOUR;

        let normalized_timezone = if t.timezone == r_efi::efi::UNSPECIFIED_TIMEZONE {
            -SYSTEMTIME_TIMEZONE
        } else {
            (t.timezone as i64) * SECS_IN_MINUTE as i64 - SYSTEMTIME_TIMEZONE
        };

        // Calculate the offset from 1/1/1900 at timezone -1440 min
        let epoch = localtime_epoch.checked_add_signed(normalized_timezone).unwrap();

        Some(Duration::new(epoch, t.nanosecond))
    }

    /// This algorithm is a modified version of the one described in the post:
    /// https://howardhinnant.github.io/date_algorithms.html#clive_from_days
    ///
    /// The changes are to use 1900-01-01-00:00:00 with timezone -1440 as anchor instead of UNIX
    /// epoch used in the original algorithm.
    pub(crate) const fn to_uefi(dur: &Duration, timezone: i16, daylight: u8) -> Result<Time, i16> {
        const MIN_IN_HOUR: u64 = 60;
        const MIN_IN_DAY: u64 = MIN_IN_HOUR * 24;

        // Check timezone validity
        assert!(timezone <= 1440 && timezone >= -1440);

        // Convert to seconds since 1900-01-01-00:00:00 in timezone.
        let Some(secs) = dur
            .as_secs()
            .checked_add_signed(SYSTEMTIME_TIMEZONE - (timezone as i64 * SECS_IN_MINUTE as i64))
        else {
            // If the current timezone cannot be used, find the closest timezone that will allow the
            // conversion to succeed.
            let new_tz = (dur.as_secs() / SECS_IN_MINUTE) as i16
                + (SYSTEMTIME_TIMEZONE / SECS_IN_MINUTE as i64) as i16;
            return Err(new_tz);
        };

        let days = secs / SECS_IN_DAY;
        let remaining_secs = secs % SECS_IN_DAY;

        let z = days + 693901;
        let era = z / 146097;
        let doe = z - (era * 146097);
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let mut y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };

        if m <= 2 {
            y += 1;
        }

        let hour = (remaining_secs / SECS_IN_HOUR) as u8;
        let minute = ((remaining_secs % SECS_IN_HOUR) / SECS_IN_MINUTE) as u8;
        let second = (remaining_secs % SECS_IN_MINUTE) as u8;

        // At this point, invalid time will be greater than MAX representable time. It cannot be less
        // than minimum time since we already take care of that case above.
        if y <= 9999 {
            Ok(Time {
                year: y as u16,
                month: m as u8,
                day: d as u8,
                hour,
                minute,
                second,
                nanosecond: dur.subsec_nanos(),
                timezone,
                daylight,
                pad1: 0,
                pad2: 0,
            })
        } else {
            assert!(y == 10000);
            assert!(m == 1);

            let delta = ((d - 1) as u64 * MIN_IN_DAY
                + hour as u64 * MIN_IN_HOUR
                + minute as u64
                + if second == 0 { 0 } else { 1 }) as i16;
            let new_tz = timezone + delta;

            assert!(new_tz <= 1440 && new_tz >= -1440);
            Err(new_tz)
        }
    }
}

pub(crate) mod instant_internal {
    use r_efi::protocols::timestamp;

    use super::super::helpers;
    use super::*;
    use crate::mem::MaybeUninit;
    use crate::ptr::NonNull;
    use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
    use crate::sys::helpers::mul_div_u64;

    const NS_PER_SEC: u64 = 1_000_000_000;

    pub fn timestamp_protocol() -> Option<Instant> {
        fn try_handle(handle: NonNull<crate::ffi::c_void>) -> Option<u64> {
            let protocol: NonNull<timestamp::Protocol> =
                helpers::open_protocol(handle, timestamp::PROTOCOL_GUID).ok()?;
            let mut properties: MaybeUninit<timestamp::Properties> = MaybeUninit::uninit();

            let r = unsafe { ((*protocol.as_ptr()).get_properties)(properties.as_mut_ptr()) };
            if r.is_error() {
                return None;
            }

            let freq = unsafe { properties.assume_init().frequency };
            let ts = unsafe { ((*protocol.as_ptr()).get_timestamp)() };
            Some(mul_div_u64(ts, NS_PER_SEC, freq))
        }

        static LAST_VALID_HANDLE: Atomic<*mut crate::ffi::c_void> =
            AtomicPtr::new(crate::ptr::null_mut());

        if let Some(handle) = NonNull::new(LAST_VALID_HANDLE.load(Ordering::Acquire)) {
            if let Some(ns) = try_handle(handle) {
                return Some(Instant(Duration::from_nanos(ns)));
            }
        }

        if let Ok(handles) = helpers::locate_handles(timestamp::PROTOCOL_GUID) {
            for handle in handles {
                if let Some(ns) = try_handle(handle) {
                    LAST_VALID_HANDLE.store(handle.as_ptr(), Ordering::Release);
                    return Some(Instant(Duration::from_nanos(ns)));
                }
            }
        }

        None
    }

    pub fn platform_specific() -> Option<Instant> {
        cfg_select! {
            any(target_arch = "x86_64", target_arch = "x86") => timestamp_rdtsc().map(Instant),
            _ => None,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn timestamp_rdtsc() -> Option<Duration> {
        static FREQUENCY: crate::sync::OnceLock<u64> = crate::sync::OnceLock::new();

        // Get Frequency in Mhz
        // Inspired by [`edk2/UefiCpuPkg/Library/CpuTimerLib/CpuTimerLib.c`](https://github.com/tianocore/edk2/blob/master/UefiCpuPkg/Library/CpuTimerLib/CpuTimerLib.c)
        let freq = FREQUENCY
            .get_or_try_init(|| {
                let cpuid = crate::arch::x86_64::__cpuid(0x15);
                if cpuid.eax == 0 || cpuid.ebx == 0 || cpuid.ecx == 0 {
                    return Err(());
                }
                Ok(mul_div_u64(cpuid.ecx as u64, cpuid.ebx as u64, cpuid.eax as u64))
            })
            .ok()?;

        let ts = unsafe { crate::arch::x86_64::_rdtsc() };
        let ns = mul_div_u64(ts, 1000, *freq);
        Some(Duration::from_nanos(ns))
    }

    #[cfg(target_arch = "x86")]
    fn timestamp_rdtsc() -> Option<Duration> {
        static FREQUENCY: crate::sync::OnceLock<u64> = crate::sync::OnceLock::new();

        let freq = FREQUENCY
            .get_or_try_init(|| {
                let cpuid = crate::arch::x86::__cpuid(0x15);
                if cpuid.eax == 0 || cpuid.ebx == 0 || cpuid.ecx == 0 {
                    return Err(());
                }
                Ok(mul_div_u64(cpuid.ecx as u64, cpuid.ebx as u64, cpuid.eax as u64))
            })
            .ok()?;

        let ts = unsafe { crate::arch::x86::_rdtsc() };
        let ns = mul_div_u64(ts, 1000, *freq);
        Some(Duration::from_nanos(ns))
    }
}
