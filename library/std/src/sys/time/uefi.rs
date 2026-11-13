use crate::sys::pal::system_time;
use crate::time::{Duration, Instant};

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
        match system_time::from_uefi(&t) {
            Some(x) => Some(Self(x)),
            None => None,
        }
    }

    pub(crate) const fn to_uefi(
        self,
        timezone: i16,
        daylight: u8,
    ) -> Result<r_efi::efi::Time, i16> {
        // system_time::to_uefi requires a valid timezone. In case of unspecified timezone,
        // we just pass 0 since it is assumed that no timezone related adjustments are required.
        if timezone == r_efi::efi::UNSPECIFIED_TIMEZONE {
            system_time::to_uefi(&self.0, 0, daylight)
        } else {
            system_time::to_uefi(&self.0, timezone, daylight)
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
        Self::from_uefi(system_time::now()).expect("time incorrectly implemented on this platform")
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

mod instant_internal {
    use core::num::niche_types::Nanoseconds;

    use r_efi::protocols::timestamp;

    use super::*;
    use crate::mem::MaybeUninit;
    use crate::ptr::NonNull;
    use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
    use crate::sys::pal::helpers;

    const NS_PER_SEC: u64 = 1_000_000_000;

    pub fn timestamp_protocol() -> Option<Instant> {
        fn try_handle(handle: NonNull<crate::ffi::c_void>) -> Option<Instant> {
            let protocol: NonNull<timestamp::Protocol> =
                helpers::open_protocol(handle, timestamp::PROTOCOL_GUID).ok()?;
            let mut properties: MaybeUninit<timestamp::Properties> = MaybeUninit::uninit();

            let r = unsafe { ((*protocol.as_ptr()).get_properties)(properties.as_mut_ptr()) };
            if r.is_error() {
                return None;
            }

            let freq = unsafe { properties.assume_init().frequency };
            let ts = unsafe { ((*protocol.as_ptr()).get_timestamp)() };

            let secs = (ts / freq) as i64;
            let subsec_ts = ts % freq;

            let nanos =
                Nanoseconds::new((subsec_ts.widening_mul(NS_PER_SEC) / u128::from(freq)) as u32)
                    .unwrap();
            Some(Instant { secs, nanos })
        }

        static LAST_VALID_HANDLE: Atomic<*mut crate::ffi::c_void> =
            AtomicPtr::new(crate::ptr::null_mut());

        if let Some(handle) = NonNull::new(LAST_VALID_HANDLE.load(Ordering::Acquire)) {
            if let Some(time) = try_handle(handle) {
                return Some(time);
            }
        }

        if let Ok(handles) = helpers::locate_handles(timestamp::PROTOCOL_GUID) {
            for handle in handles {
                if let Some(time) = try_handle(handle) {
                    LAST_VALID_HANDLE.store(handle.as_ptr(), Ordering::Release);
                    return Some(time);
                }
            }
        }

        None
    }

    pub fn platform_specific() -> Option<Instant> {
        cfg_select! {
            any(target_arch = "x86", target_arch = "x86_64") => timestamp_rdtsc(),
            _ => None,
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn timestamp_rdtsc() -> Option<Instant> {
        #[cfg(target_arch = "x86")]
        use crate::arch::x86::{__cpuid, _rdtsc};
        #[cfg(target_arch = "x86_64")]
        use crate::arch::x86_64::{__cpuid, _rdtsc};

        struct Slope {
            multiplier: u32,
            divisor: u64,
        }

        static SLOPE: crate::sync::OnceLock<Slope> = crate::sync::OnceLock::new();

        // Inspired by https://github.com/tianocore/edk2/blob/6d127c21406c89b50f6c7345f02c2b958660e231/UefiCpuPkg/Library/CpuTimerLib/CpuTimerLib.c
        let slope = SLOPE
            .get_or_try_init(|| {
                let cpuid = __cpuid(0x15);
                if cpuid.eax == 0 || cpuid.ebx == 0 || cpuid.ecx == 0 {
                    return Err(());
                }

                let core_freq_mhz = cpuid.ecx;
                let core_freq_tsc_mul = cpuid.ebx;
                let core_freq_tsc_div = cpuid.eax;

                // TSC_freq_hz = (core_freq_hz * core_freq_tsc_mul) / core_freq_tsc_div
                // time = TSC / TSC_freq_hz
                //      = TSC / ((core_freq_hz * core_freq_tsc_mul) / core_freq_tsc_div)
                //      = (TSC * core_freq_tsc_div) / (core_freq_hz * core_freq_tsc_mul)

                Ok(Slope {
                    multiplier: core_freq_tsc_div,
                    divisor: core_freq_mhz.widening_mul(core_freq_tsc_mul),
                })
            })
            .ok()?;

        let ts = unsafe { _rdtsc() };

        let divisor = u128::from(slope.divisor);
        let numerator = ts.widening_mul(u64::from(slope.multiplier));
        // The TSC definitely runs faster than 2 Hz, hence this cannot overflow.
        let secs = (numerator / divisor) as i64;
        let remainder = (numerator % divisor) as u64;
        let nanos =
            Nanoseconds::new((remainder.widening_mul(NS_PER_SEC) / divisor) as u32).unwrap();

        Some(Instant { secs, nanos })
    }
}
