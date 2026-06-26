use crate::sys::pal::system_time;
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
    use r_efi::protocols::timestamp;

    use super::*;
    use crate::mem::MaybeUninit;
    use crate::ptr::NonNull;
    use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
    use crate::sys::helpers::mul_div_u64;
    use crate::sys::pal::helpers;

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
