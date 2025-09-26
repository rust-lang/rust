use crate::time::{Duration, Instant};

pub fn sleep(dur: Duration) {
    // Sleep in increments of `u64::MAX` nanoseconds until the `dur` is
    // entirely drained.
    let mut remaining = dur.as_nanos();
    while remaining > 0 {
        let amt = u64::try_from(remaining).unwrap_or(u64::MAX);
        wasip2::clocks::monotonic_clock::subscribe_duration(amt).block();
        remaining -= u128::from(amt);
    }
}

pub fn sleep_until(deadline: Instant) {
    wasip2::clocks::monotonic_clock::subscribe_instant(deadline.into_inner().to_wasi_instant())
        .block();
}
