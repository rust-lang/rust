use crate::ops::Neg;
use crate::ptr::null;
use crate::sys::pal::c;
use crate::time::Duration;

const NANOS_PER_SEC: u64 = 1_000_000_000;
pub const INTERVALS_PER_SEC: u64 = NANOS_PER_SEC / 100;

pub fn checked_dur2intervals(dur: &Duration) -> Option<i64> {
    dur.as_secs()
        .checked_mul(INTERVALS_PER_SEC)?
        .checked_add(dur.subsec_nanos() as u64 / 100)?
        .try_into()
        .ok()
}

pub fn intervals2dur(intervals: u64) -> Duration {
    Duration::new(intervals / INTERVALS_PER_SEC, ((intervals % INTERVALS_PER_SEC) * 100) as u32)
}

pub mod perf_counter {
    use super::NANOS_PER_SEC;
    use crate::sync::atomic::{AtomicI64, Ordering};
    use crate::sys::c;
    use crate::sys::helpers::mul_div_u64;
    use crate::time::Duration;

    pub fn convert(counter: Counter, frequency: Frequency, magnitude: u64) -> u64 {
        let counter = counter.as_u64();
        let freq = frequency.as_u64();
        // This looks redundant but it allows the optimizer to optimize
        // for common values 10_000_000 (on x86/x64) and 24_000_000 (on aarch64).
        // Note: the `10_000_000` branch is never taken on aarch64 but it's necessary
        // to improve codegen. See #162074.
        if freq == 10_000_000 {
            mul_div_u64(counter, magnitude, freq)
        } else if freq == 24_000_000 && cfg!(target_arch = "aarch64") {
            mul_div_u64(counter, magnitude, freq)
        } else {
            mul_div_u64(counter, magnitude, freq)
        }
    }

    type ValidCounter = core::num::niche_types::PositiveI64;
    #[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
    #[repr(transparent)]
    pub struct Counter {
        value: ValidCounter,
    }
    impl Counter {
        pub fn query() -> Self {
            let mut qpc_value: i64 = 0;
            // `QueryPerformanceCounter` will never fail (since XP).
            unsafe { c::QueryPerformanceCounter(&mut qpc_value) };
            // SAFETY: qpc_value will always be a positive integer since QueryPerformanceCounter
            // "reads the performance counter and returns the total number of ticks that have
            // occurred since the Windows operating system was started"
            // See: https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
            unsafe { Self::new_unchecked(qpc_value) }
        }

        pub fn checked_sub(self, other: Self) -> Option<Self> {
            self.as_u64()
                .checked_sub(other.as_u64())
                // SAFETY: unsigned `checked_sub` ensure that self >= other and both
                // are positive integers <=i64::MAX so the result will also
                // be in the range 0..=i64::MAX.
                .map(|n| unsafe { Self::new_unchecked(n as i64) })
        }

        pub fn as_u64(self) -> u64 {
            // self always holds a positive i64 value.
            self.value.as_inner() as u64
        }

        /// Construct a Counter from the result of `QueryPerformanceCounter`
        ///
        /// # Safety
        ///
        /// `freq` must be a positive integer.
        /// The result of `QueryPerformanceCounter` fulfills this requirement.
        unsafe fn new_unchecked(freq: i64) -> Self {
            // SAFETY: The caller must ensure this is safe.
            unsafe { Self { value: ValidCounter::new_unchecked(freq) } }
        }
    }

    // This type helps the compiler to optimize calculations involving QueryPerformanceFrequency.
    type ValidFrequency = core::num::niche_types::PositiveNonZeroI64;
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct Frequency {
        value: ValidFrequency,
    }

    impl Frequency {
        pub fn query() -> Self {
            let freq = frequency();
            // SAFETY: According to the MSDN entry for `QueryPerformanceFrequency`,
            // a value of 0 will never be returned starting from Windows XP.
            // A frequency will also always be a positive value.
            unsafe { Self::new_unchecked(freq) }
        }

        fn as_u64(self) -> u64 {
            // self always holds a positive i64 value.
            self.value.as_inner() as u64
        }

        /// Construct a Frequency from the result of `QueryPerformanceFrequency`
        ///
        /// # Safety
        ///
        /// `freq` must be a positive integer greater than 0.
        /// The result of `QueryPerformanceFrequency` fulfills this requirement (since XP).
        unsafe fn new_unchecked(freq: i64) -> Self {
            // SAFETY: The caller must ensure this is safe.
            unsafe { Self { value: ValidFrequency::new_unchecked(freq) } }
        }
    }

    fn frequency() -> i64 {
        // Either the cached result of `QueryPerformanceFrequency` or `0` for
        // uninitialized. Storing this as a single `AtomicI64` allows us to use
        // `Relaxed` operations, as we are only interested in the effects on a
        // single memory location.
        static FREQUENCY: AtomicI64 = AtomicI64::new(0);

        let cached = FREQUENCY.load(Ordering::Relaxed);
        // If a previous thread has filled in this global state, use that.
        if cached != 0 {
            return cached;
        }
        // ... otherwise learn for ourselves ...
        frequency_init(&FREQUENCY)
    }

    #[cold]
    fn frequency_init(cache: &AtomicI64) -> i64 {
        let mut frequency = 0;
        // `QueryPerformanceFrequency` will never fail (since XP).
        // SAFETY: it just writes to frequency.
        unsafe { c::QueryPerformanceFrequency(&mut frequency) };

        cache.store(frequency, Ordering::Relaxed);
        frequency
    }

    // Per microsoft docs, the margin of error for cross-thread time comparisons
    // using QueryPerformanceCounter is 1 "tick" -- defined as 1/frequency().
    // Reference: https://docs.microsoft.com/en-us/windows/desktop/SysInfo
    //                   /acquiring-high-resolution-time-stamps
    pub fn epsilon() -> Duration {
        let epsilon = NANOS_PER_SEC / (frequency() as u64);
        Duration::from_nanos(epsilon)
    }
}

/// A timer you can wait on.
pub(crate) struct WaitableTimer {
    handle: c::HANDLE,
}

impl WaitableTimer {
    /// Creates a high-resolution timer. Will fail before Windows 10, version 1803.
    pub fn high_resolution() -> Result<Self, ()> {
        let handle = unsafe {
            c::CreateWaitableTimerExW(
                null(),
                null(),
                c::CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                c::TIMER_ALL_ACCESS,
            )
        };
        if !handle.is_null() { Ok(Self { handle }) } else { Err(()) }
    }

    pub fn set(&self, duration: Duration) -> Result<(), ()> {
        // Convert the Duration to a format similar to FILETIME.
        // Negative values are relative times whereas positive values are absolute.
        // Therefore we negate the relative duration.
        let time = checked_dur2intervals(&duration).ok_or(())?.neg();
        let result = unsafe { c::SetWaitableTimer(self.handle, &time, 0, None, null(), c::FALSE) };
        if result != 0 { Ok(()) } else { Err(()) }
    }

    pub fn wait(&self) -> Result<(), ()> {
        let result = unsafe { c::WaitForSingleObject(self.handle, c::INFINITE) };
        if result != c::WAIT_FAILED { Ok(()) } else { Err(()) }
    }
}

impl Drop for WaitableTimer {
    fn drop(&mut self) {
        unsafe { c::CloseHandle(self.handle) };
    }
}
