use crate::sys::time;

#[inline]
pub(super) fn monotonize(raw: time::Instant) -> time::Instant {
    inner::monotonize(raw)
}

#[cfg(any(all(target_has_atomic = "64", not(target_has_atomic = "128")), target_arch = "aarch64"))]
pub mod inner {
    use crate::sync::atomic::AtomicU64;
    use crate::sync::atomic::Ordering::*;
    use crate::sys::time;
    use crate::time::Duration;

    pub(in crate::time) const ZERO: time::Instant = time::Instant::zero();

    // bits 30 and 31 are never used since the nanoseconds part never exceeds 10^9
    const UNINITIALIZED: u64 = 0b11 << 30;
    static MONO: AtomicU64 = AtomicU64::new(UNINITIALIZED);

    #[inline]
    pub(super) fn monotonize(raw: time::Instant) -> time::Instant {
        monotonize_impl(&MONO, raw)
    }

    #[inline]
    pub(in crate::time) fn monotonize_impl(mono: &AtomicU64, raw: time::Instant) -> time::Instant {
        let delta = raw.checked_sub_instant(&ZERO).unwrap();
        let secs = delta.as_secs();
        // occupies no more than 30 bits (10^9 seconds)
        let nanos = delta.subsec_nanos() as u64;

        // This wraps around every 136 years (2^32 seconds).
        // To detect backsliding we use wrapping arithmetic and declare forward steps smaller
        // than 2^31 seconds as expected and everything else as a backslide which will be
        // monotonized.
        // This could be a problem for programs that call instants at intervals greater
        // than 68 years. Interstellar probes may want to ensure that actually_monotonic() is true.
        let packed = (secs << 32) | nanos;
        let updated = mono.fetch_update(Relaxed, Relaxed, |old| {
            (old == UNINITIALIZED || packed.wrapping_sub(old) < u64::MAX / 2).then_some(packed)
        });
        match updated {
            Ok(_) => raw,
            Err(newer) => {
                // Backslide occurred. We reconstruct monotonized time from the upper 32 bit of the
                // passed in value and the 64bits loaded from the atomic
                let seconds_lower = newer >> 32;
                let mut seconds_upper = secs & 0xffff_ffff_0000_0000;
                if secs & 0xffff_ffff > seconds_lower {
                    // Backslide caused the lower 32bit of the seconds part to wrap.
                    // This must be the case because the seconds part is larger even though
                    // we are in the backslide branch, i.e. the seconds count should be smaller or equal.
                    //
                    // We assume that backslides are smaller than 2^32 seconds
                    // which means we need to add 1 to the upper half to restore it.
                    //
                    // Example:
                    // most recent observed time: 0xA1_0000_0000_0000_0000u128
                    // bits stored in AtomicU64:     0x0000_0000_0000_0000u64
                    // backslide by 1s
                    // caller time is             0xA0_ffff_ffff_0000_0000u128
                    // -> we can fix up the upper half time by adding 1 << 32
                    seconds_upper = seconds_upper.wrapping_add(0x1_0000_0000);
                }
                let secs = seconds_upper | seconds_lower;
                let nanos = newer as u32;
                ZERO.checked_add_duration(&Duration::new(secs, nanos)).unwrap()
            }
        }
    }
}

#[cfg(all(target_has_atomic = "128", not(target_arch = "aarch64")))]
pub mod inner {
    use crate::sync::atomic::AtomicU128;
    use crate::sync::atomic::Ordering::*;
    use crate::sys::time;
    use crate::time::Duration;

    const ZERO: time::Instant = time::Instant::zero();
    static MONO: AtomicU128 = AtomicU128::new(0);

    #[inline]
    pub(super) fn monotonize(raw: time::Instant) -> time::Instant {
        let delta = raw.checked_sub_instant(&ZERO).unwrap();
        // Split into seconds and nanos since Duration doesn't have a
        // constructor that takes a u128
        let secs = delta.as_secs() as u128;
        let nanos = delta.subsec_nanos() as u128;
        let timestamp: u128 = secs << 64 | nanos;
        let timestamp = MONO.fetch_max(timestamp, Relaxed).max(timestamp);
        let secs = (timestamp >> 64) as u64;
        let nanos = timestamp as u32;
        ZERO.checked_add_duration(&Duration::new(secs, nanos)).unwrap()
    }
}

#[cfg(not(any(target_has_atomic = "64", target_has_atomic = "128")))]
pub mod inner {
    use crate::cmp;
    use crate::sys::time;
    use crate::sys_common::mutex::StaticMutex;

    #[inline]
    pub(super) fn monotonize(os_now: time::Instant) -> time::Instant {
        static LOCK: StaticMutex = StaticMutex::new();
        static mut LAST_NOW: time::Instant = time::Instant::zero();
        unsafe {
            let _lock = LOCK.lock();
            let now = cmp::max(LAST_NOW, os_now);
            LAST_NOW = now;
            now
        }
    }
}
