use crate::sys::helpers::mul_div_u64;
use crate::sys::pal::time::mach_timebase_info;
use crate::time::Duration;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Stopwatch {
    ticks: u64,
}

impl Stopwatch {
    pub fn start() -> Self {
        unsafe extern "C" {
            // SAFETY: it only retrieves ticks as a u64
            pub safe fn mach_absolute_time() -> u64;
        }
        Self { ticks: mach_absolute_time() }
    }

    pub fn checked_duration_since(self, other: Stopwatch) -> Option<Duration> {
        let diff = self.ticks.checked_sub(other.ticks)?;
        let timebase = mach_timebase_info();
        let nanos = mul_div_u64(diff, timebase.numer as u64, timebase.denom as u64);
        Some(Duration::from_nanos(nanos))
    }
}
