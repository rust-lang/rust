use crate::time::Duration;

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
});

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
});

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
    pub(crate) const fn from_uefi(t: r_efi::efi::Time) -> Self {
        Self(system_time_internal::from_uefi(&t))
    }

    #[expect(dead_code)]
    pub(crate) const fn to_uefi(self, timezone: i16, daylight: u8) -> Option<r_efi::efi::Time> {
        system_time_internal::to_uefi(&self.0, timezone, daylight)
    }

    pub fn now() -> SystemTime {
        system_time_internal::now()
            .unwrap_or_else(|| panic!("time not implemented on this platform"))
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

    const SECS_IN_MINUTE: u64 = 60;
    const SECS_IN_HOUR: u64 = SECS_IN_MINUTE * 60;
    const SECS_IN_DAY: u64 = SECS_IN_HOUR * 24;
    const TIMEZONE_DELTA: u64 = 1440 * SECS_IN_MINUTE;

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

        Some(SystemTime::from_uefi(t))
    }

    /// This algorithm is a modified form of the one described in the post
    /// https://blog.reverberate.org/2020/05/12/optimizing-date-algorithms.html
    ///
    /// The changes are to use 1900-01-01-00:00:00 with timezone -1440 as anchor instead of UNIX
    /// epoch used in the original algorithm.
    pub(crate) const fn from_uefi(t: &Time) -> Duration {
        assert!(t.month <= 12 && t.month != 0);
        assert!(t.year >= 1900 && t.year <= 9999);
        assert!(t.day <= 31 && t.day != 0);

        assert!(t.second < 60);
        assert!(t.minute < 60);
        assert!(t.hour < 24);
        assert!(t.nanosecond < 1_000_000_000);

        assert!(
            (t.timezone <= 1440 && t.timezone >= -1440)
                || t.timezone == r_efi::efi::UNSPECIFIED_TIMEZONE
        );

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

        // Calculate the offset from 1/1/1900 at timezone -1440 min
        let adjusted_localtime_epoc: u64 = localtime_epoch + TIMEZONE_DELTA;

        let epoch: u64 = if t.timezone == r_efi::efi::UNSPECIFIED_TIMEZONE {
            adjusted_localtime_epoc
        } else {
            adjusted_localtime_epoc
                .checked_add_signed((t.timezone as i64) * SECS_IN_MINUTE as i64)
                .unwrap()
        };

        Duration::new(epoch, t.nanosecond)
    }

    /// This algorithm is a modified version of the one described in the post:
    /// https://howardhinnant.github.io/date_algorithms.html#clive_from_days
    ///
    /// The changes are to use 1900-01-01-00:00:00 with timezone -1440 as anchor instead of UNIX
    /// epoch used in the original algorithm.
    pub(crate) const fn to_uefi(dur: &Duration, timezone: i16, daylight: u8) -> Option<Time> {
        // Check timzone validity
        assert!(timezone <= 1440 && timezone >= -1440);

        // FIXME(#126043): use checked_sub_signed once stabilized
        let secs =
            dur.as_secs().checked_add_signed((-timezone as i64) * SECS_IN_MINUTE as i64).unwrap();

        // Convert to seconds since 1900-01-01-00:00:00 in timezone.
        let Some(secs) = secs.checked_sub(TIMEZONE_DELTA) else { return None };

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

        // Check Bounds
        if y >= 1900 && y <= 9999 {
            Some(Time {
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
            None
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
    use crate::sys_common::mul_div_u64;

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
                let cpuid = unsafe { crate::arch::x86_64::__cpuid(0x15) };
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
                let cpuid = unsafe { crate::arch::x86::__cpuid(0x15) };
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
