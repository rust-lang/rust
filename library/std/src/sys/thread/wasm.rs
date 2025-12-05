use crate::cmp;
use crate::time::Duration;

pub fn sleep(dur: Duration) {
    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32 as wasm;
    #[cfg(target_arch = "wasm64")]
    use core::arch::wasm64 as wasm;

    // Use an atomic wait to block the current thread artificially with a
    // timeout listed. Note that we should never be notified (return value
    // of 0) or our comparison should never fail (return value of 1) so we
    // should always only resume execution through a timeout (return value
    // 2).
    let mut nanos = dur.as_nanos();
    while nanos > 0 {
        let amt = cmp::min(i64::MAX as u128, nanos);
        let mut x = 0;
        let val = unsafe { wasm::memory_atomic_wait32(&mut x, 0, amt as i64) };
        debug_assert_eq!(val, 2);
        nanos -= amt;
    }
}
