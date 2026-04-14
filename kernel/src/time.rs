use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub static SYSTEM_TIME_OFFSET: AtomicU64 = AtomicU64::new(0);
static IS_ANCHORED: AtomicBool = AtomicBool::new(false);
static MONOTONIC_CLAMP: MonotonicClamp = MonotonicClamp::new();

pub const NANOS_PER_SEC: u64 = 1_000_000_000;
pub const SCHED_TICK_HZ: u64 = 100;
pub const SCHED_TICK_NANOS: u64 = NANOS_PER_SEC / SCHED_TICK_HZ;

/// Returns true if the system clock has been anchored to a wall-clock time source.
pub fn is_anchored() -> bool {
    IS_ANCHORED.load(Ordering::Relaxed)
}

/// Returns system time in nanoseconds (monotonic + offset)
pub fn get_system_time_ns(mono_ns: u64) -> u64 {
    mono_ns.saturating_add(SYSTEM_TIME_OFFSET.load(Ordering::Relaxed))
}

/// Set the system time offset directly (in nanoseconds)
pub fn set_system_time_offset(offset_ns: u64) {
    SYSTEM_TIME_OFFSET.store(offset_ns, Ordering::Relaxed);
}

/// Anchor system clock: given a Unix timestamp and the corresponding monotonic time,
/// compute and store the offset so that future time queries return correct wall-clock time.
pub fn anchor_system_clock(unix_secs: u64, mono_ns: u64) {
    let unix_ns = unix_secs.saturating_mul(NANOS_PER_SEC);
    let offset = unix_ns.saturating_sub(mono_ns);
    set_system_time_offset(offset);
    IS_ANCHORED.store(true, Ordering::Relaxed);
    crate::kinfo!(
        "System clock anchored: unix_secs={}, mono_ns={}, offset={}ns",
        unix_secs,
        mono_ns,
        offset
    );
}

/// Return the current wall-clock time as `(seconds, nanoseconds)`.
///
/// If the runtime has not yet been initialized (e.g. during early boot or
/// in unit tests), or if the clock has not been anchored to a real-time
/// source, returns `(0, 0)` — which corresponds to the Unix epoch.
/// Callers that need precise wall-clock time should check [`is_anchored`] first.
///
/// This is the canonical time source for VFS node timestamps.
pub fn now_timespec() -> (u64, u32) {
    let mono_ns = monotonic_now_ns();
    let sys_ns = if is_anchored() {
        get_system_time_ns(mono_ns)
    } else {
        0
    };
    let sec = sys_ns / NANOS_PER_SEC;
    let nsec = (sys_ns % NANOS_PER_SEC) as u32;
    (sec, nsec)
}

pub fn runtime_ticks_to_nanos(ticks: u64, freq_hz: u64) -> u64 {
    if freq_hz == 0 {
        return 0;
    }
    ((ticks as u128).saturating_mul(NANOS_PER_SEC as u128) / freq_hz as u128) as u64
}

pub fn monotonic_now_ns() -> u64 {
    if !crate::is_runtime_initialized() {
        return 0;
    }
    let rt = crate::runtime_base();
    let raw = runtime_ticks_to_nanos(rt.mono_ticks(), rt.mono_freq_hz());
    MONOTONIC_CLAMP.clamp(raw)
}

pub fn realtime_now_ns() -> Option<u64> {
    if !is_anchored() {
        return None;
    }
    Some(get_system_time_ns(monotonic_now_ns()))
}

pub fn duration_to_sleep_ticks(duration_ns: u64) -> u64 {
    if duration_ns == 0 {
        0
    } else {
        duration_ns.saturating_add(SCHED_TICK_NANOS - 1) / SCHED_TICK_NANOS
    }
}

pub struct MonotonicClamp {
    last: AtomicU64,
}

impl MonotonicClamp {
    pub const fn new() -> Self {
        Self {
            last: AtomicU64::new(0),
        }
    }

    pub fn clamp(&self, raw: u64) -> u64 {
        let last = self.last.load(Ordering::Relaxed);
        if raw > last {
            if let Err(actual) =
                self.last
                    .compare_exchange(last, raw, Ordering::Relaxed, Ordering::Relaxed)
            {
                if actual > raw { actual } else { raw }
            } else {
                raw
            }
        } else {
            last
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MonotonicClamp, NANOS_PER_SEC, SCHED_TICK_NANOS, duration_to_sleep_ticks,
        runtime_ticks_to_nanos,
    };

    #[test]
    fn runtime_ticks_to_nanos_handles_zero_frequency() {
        assert_eq!(runtime_ticks_to_nanos(123, 0), 0);
    }

    #[test]
    fn runtime_ticks_to_nanos_converts_fractional_seconds() {
        assert_eq!(runtime_ticks_to_nanos(50, 100), NANOS_PER_SEC / 2);
    }

    #[test]
    fn sleep_tick_rounding_is_ceiling() {
        assert_eq!(duration_to_sleep_ticks(0), 0);
        assert_eq!(duration_to_sleep_ticks(1), 1);
        assert_eq!(duration_to_sleep_ticks(SCHED_TICK_NANOS), 1);
        assert_eq!(duration_to_sleep_ticks(SCHED_TICK_NANOS + 1), 2);
    }

    #[test]
    fn monotonic_clamp_never_goes_backwards() {
        let clamp = MonotonicClamp::new();
        assert_eq!(clamp.clamp(100), 100);
        assert_eq!(clamp.clamp(90), 100);
        assert_eq!(clamp.clamp(110), 110);
    }
}
