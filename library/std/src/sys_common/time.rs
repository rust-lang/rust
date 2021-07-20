use cfg_if::cfg_if;

#[cfg(test)]
mod tests;

#[cfg_attr(not(windows), allow(dead_code))]
pub const MONOTONIZE_U64: bool = cfg!(target_has_atomic = "64");

#[cfg_attr(not(windows), allow(dead_code))]
pub fn monotonize_u64(tick: u64) -> Option<u64> {
    if MONOTONIZE_U64 {
        use crate::sync::atomic::AtomicU64;
        use crate::sync::atomic::Ordering::Relaxed;

        static LAST_NOW: AtomicU64 = AtomicU64::new(0);

        return Some(LAST_NOW.fetch_max(tick, Relaxed).max(tick));
    }

    None
}

#[cfg_attr(any(not(unix), target_os = "ios", target_os = "macos"), allow(dead_code))]
pub fn can_monotonize_u64x2() -> bool {
    cfg!(target_has_atomic = "128") || have_cmpxchg16b()
}

#[cfg_attr(any(not(unix), target_os = "ios", target_os = "macos"), allow(dead_code))]
fn have_cmpxchg16b() -> bool {
    cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            return std_detect::is_x86_feature_detected!("cmpxchg16b");
        } else {
            return false
        }
    }
}

#[cfg_attr(any(not(unix), target_os = "ios", target_os = "macos"), allow(dead_code))]
#[allow(unused_variables)]
pub fn monotonize_u64x2(sec: u64, nsec: u64) -> Option<(u64, u64)> {
    #[cfg(target_arch = "x86_64")]
    if have_cmpxchg16b() {
        let os_now = (sec as u128) << 64 | nsec as u128;
        let monotonized = unsafe { monotonize_cmpxchg16b(os_now) };
        return Some(((monotonized >> 64) as u64, monotonized as u64));
    }

    cfg_if! {
        if #[cfg(target_has_atomic = "128")] {
            use core::sync::atomic::AtomicU128;
            use crate::sync::atomic::Ordering::Relaxed;

            static LAST_NOW: AtomicU128 = AtomicU128::new(0);

            let os_now = (sec as u128) << 64 | nsec as u128;
            let monotonized = LAST_NOW.fetch_max(os_now, Relaxed).max(os_now);
            return Some(((monotonized >> 64) as u64, monotonized as u64));
        } else {
            return None;
        }
    }
}

#[cfg_attr(any(not(unix), target_os = "ios", target_os = "macos"), allow(dead_code))]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "cmpxchg16b")]
// FIXME(#56798) inlining this on baseline x86_64 would fail to link __sync_val_compare_and_swap_16
#[cfg_attr(not(target_feature = "cmpxchg16b"), inline(never))]
unsafe fn monotonize_cmpxchg16b(os_now: u128) -> u128 {
    use crate::ptr::addr_of_mut;
    use crate::sync::atomic::Ordering::Relaxed;
    use core::arch::x86_64::cmpxchg16b;
    use core::intrinsics::{atomic_load_relaxed, prefetch_write_data};
    use core::mem::transmute;

    #[repr(align(16))]
    struct AlignedInstant(u128);

    static mut LAST_NOW: AlignedInstant = AlignedInstant(0);

    let last_ptr = addr_of_mut!(LAST_NOW) as *mut u128;

    // obtain exclusive access to the cache line since we'll immediately follow up the loads
    // (which would result in shared ownership) with a cmpxchg16b that requires it.
    prefetch_write_data(last_ptr, 0);
    // Attempt a piece-wise atomic load to avoid chaining multiple CAS operations.
    // The drawback is that we may experience torn reads.
    let last_now = [
        atomic_load_relaxed(last_ptr as *const u64),
        atomic_load_relaxed((last_ptr as *const u64).add(1)),
    ];

    let mut last_now: u128 = transmute::<_, u128>(last_now);

    loop {
        // ideally we would immediately return last_now when it is larger than os_now
        // but we cannot do that since it may be a torn read so we still have to
        // perform at least one CAS.
        let now_max = crate::cmp::max(os_now, last_now);

        let result = cmpxchg16b(last_ptr, last_now, now_max, Relaxed, Relaxed);

        // CAS succeeded
        if result == last_now {
            return now_max;
        }

        // CAS failed and we obtained a newer instant from it
        if result >= os_now {
            return result;
        }

        // CAS failed but the time is not newer than what we have. this likely is due
        // to a torn read; retry.
        // This can happen at most once since the new read isn't torn anymore so
        // the next iteration will either succeed or obtain a newer value
        last_now = result;
    }
}
