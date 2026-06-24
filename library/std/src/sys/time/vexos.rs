use core::num::niche_types::Nanoseconds;

use crate::time::Instant;

pub fn now() -> Instant {
    let micros = unsafe { vex_sdk::vexSystemHighResTimeGet() };
    let secs = (micros / 1_000_000) as i64;
    let nanos = Nanoseconds::new(1000 * (micros % 1_000_000) as u32).unwrap();
    Instant { secs, nanos }
}
