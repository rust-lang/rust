use super::common;
use crate::mem::MaybeUninit;
use crate::ptr::NonNull;
use crate::sys_common::mul_div_u64;
use crate::time::Duration;

use r_efi::protocols::timestamp;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SystemTime(Duration);

pub const UNIX_EPOCH: SystemTime = SystemTime(Duration::ZERO);

const NS_PER_SEC: u64 = 1_000_000_000;
const SEC_IN_MIN: u64 = 60;
const SEC_IN_HOUR: u64 = SEC_IN_MIN * 60;
const SEC_IN_DAY: u64 = SEC_IN_HOUR * 24;
const SEC_IN_YEAR: u64 = SEC_IN_DAY * 365;

impl Instant {
    pub fn now() -> Instant {
        if let Ok(handles) = common::locate_handles(timestamp::PROTOCOL_GUID) {
            // First try using `EFI_TIMESTAMP_PROTOCOL` if present
            for handle in handles {
                let protocol: NonNull<timestamp::Protocol> =
                    match common::open_protocol(handle, timestamp::PROTOCOL_GUID) {
                        Ok(x) => x,
                        Err(_) => continue,
                    };
                let mut properties: MaybeUninit<timestamp::Properties> = MaybeUninit::uninit();
                let r = unsafe { ((*protocol.as_ptr()).get_properties)(properties.as_mut_ptr()) };
                if r.is_error() {
                    continue;
                } else {
                    let properties = unsafe { properties.assume_init() };
                    let ts = unsafe { ((*protocol.as_ptr()).get_timestamp)() };
                    let frequency = properties.frequency;
                    let ns = mul_div_u64(ts, NS_PER_SEC, frequency);
                    return Instant(Duration::from_nanos(ns));
                }
            }
        }

        // Try using raw CPU Registers
        // Currently only implemeted for x86_64 using CPUID (0x15) and TSC register
        #[cfg(target_arch = "x86_64")]
        if let Some(ns) = get_timestamp() {
            return Instant(Duration::from_nanos(ns));
        }

        if let Some(runtime_services) = common::get_runtime_services() {
            // Finally just use `EFI_RUNTIME_SERVICES.GetTime()`
            let mut t = r_efi::efi::Time::default();
            let r =
                unsafe { ((*runtime_services.as_ptr()).get_time)(&mut t, crate::ptr::null_mut()) };

            if r.is_error() {
                panic!("time not implemented on this platform")
            } else {
                return Instant(uefi_time_to_duration(t));
            }
        }

        // Panic if nothing works
        panic!("Failed to create Instant")
    }

    #[inline]
    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0)
    }

    #[inline]
    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add(*other)?))
    }

    #[inline]
    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub(*other)?))
    }
}

// Using Unix representation of Time.
impl SystemTime {
    pub fn now() -> SystemTime {
        if let Some(runtime_services) = common::get_runtime_services() {
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

    #[inline]
    pub fn sub_time(&self, other: &SystemTime) -> Result<Duration, Duration> {
        self.0.checked_sub(other.0).ok_or_else(|| other.0 - self.0)
    }

    #[inline]
    pub fn checked_add_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_add(*other)?))
    }

    #[inline]
    pub fn checked_sub_duration(&self, other: &Duration) -> Option<SystemTime> {
        Some(SystemTime(self.0.checked_sub(*other)?))
    }

    pub(crate) fn get_duration(&self) -> Duration {
        self.0
    }
}

impl From<r_efi::system::Time> for SystemTime {
    #[inline]
    fn from(t: r_efi::system::Time) -> Self {
        SystemTime(uefi_time_to_duration(t))
    }
}

// FIXME: Don't know how to use Daylight Saving thing
fn uefi_time_to_duration(t: r_efi::system::Time) -> Duration {
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

// This algorithm is taken from: http://howardhinnant.github.io/date_algorithms.html
pub(crate) fn uefi_time_from_duration(
    dur: Duration,
    daylight: u8,
    timezone: i16,
) -> r_efi::system::Time {
    let secs = dur.as_secs();

    let days = secs / SEC_IN_DAY;
    let remaining_secs = secs % SEC_IN_DAY;

    let z = days + 719468;
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

    r_efi::system::Time {
        year: y as u16,
        month: m as u8,
        day: d as u8,
        hour: (remaining_secs / SEC_IN_HOUR) as u8,
        minute: ((remaining_secs % SEC_IN_HOUR) / SEC_IN_MIN) as u8,
        second: ((remaining_secs % SEC_IN_HOUR) % SEC_IN_MIN) as u8,
        pad1: 0,
        nanosecond: dur.subsec_nanos(),
        timezone,
        daylight,
        pad2: 0,
    }
}

// Returns the Frequency in Mhz
// Mostly based on [`edk2/UefiCpuPkg/Library/CpuTimerLib/CpuTimerLib.c`](https://github.com/tianocore/edk2/blob/master/UefiCpuPkg/Library/CpuTimerLib/CpuTimerLib.c)
// Currently implemented only for x86_64 but can be extended for other arch if they ever support
// std.
#[cfg(target_arch = "x86_64")]
fn frequency() -> Option<u64> {
    use crate::sync::atomic::{AtomicU64, Ordering};

    static FREQUENCY: AtomicU64 = AtomicU64::new(0);

    let cached = FREQUENCY.load(Ordering::Relaxed);
    if cached != 0 {
        return Some(cached);
    }

    if crate::arch::x86_64::has_cpuid() {
        let cpuid = unsafe { crate::arch::x86_64::__cpuid(0x15) };

        if cpuid.eax == 0 || cpuid.ebx == 0 || cpuid.ecx == 0 {
            return None;
        }

        let freq = mul_div_u64(cpuid.ecx as u64, cpuid.ebx as u64, cpuid.eax as u64);
        FREQUENCY.store(freq, Ordering::Relaxed);
        return Some(freq);
    }

    None
}

// Currently implemented only for x86_64 but can be extended for other arch if they ever support
// std.
#[cfg(target_arch = "x86_64")]
fn get_timestamp() -> Option<u64> {
    let freq = frequency()?;
    let ts = unsafe { crate::arch::x86_64::_rdtsc() };
    let ns = mul_div_u64(ts, 1000, freq);
    Some(ns)
}
