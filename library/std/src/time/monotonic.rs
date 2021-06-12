use crate::sys::time;

#[inline]
pub(super) fn monotonize(raw: time::Instant) -> time::Instant {
    inner::monotonize(raw)
}

#[cfg(all(target_has_atomic = "64", not(target_has_atomic = "128")))]
pub mod inner {
    use crate::sync::atomic::AtomicU64;
    use crate::sync::atomic::Ordering::*;
    use crate::sys::time;
    use crate::time::Duration;

    const ZERO: time::Instant = time::Instant::zero();

    // bits 30 and 31 are never used since the seconds part never exceeds 10^9
    const UNINITIALIZED: u64 = 0xff00_0000;
    static MONO: AtomicU64 = AtomicU64::new(UNINITIALIZED);

    #[inline]
    pub(super) fn monotonize(raw: time::Instant) -> time::Instant {
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
        let old = MONO.load(Relaxed);

        if packed == UNINITIALIZED || packed.wrapping_sub(old) < u64::MAX / 2 {
            MONO.store(packed, Relaxed);
            raw
        } else {
            // Backslide occurred. We reconstruct monotonized time by assuming the clock will never
            // backslide more than 2`32 seconds which means we can reuse the upper 32bits from
            // the seconds.
            let secs = (secs & 0xffff_ffff << 32) | old >> 32;
            let nanos = old as u32;
            ZERO.checked_add_duration(&Duration::new(secs, nanos)).unwrap()
        }
    }
}

#[cfg(target_has_atomic = "128")]
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
        // constructor that takes an u128
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
