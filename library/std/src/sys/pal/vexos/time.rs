use crate::time::Duration;

#[expect(dead_code)]
#[path = "../unsupported/time.rs"]
mod unsupported_time;
pub use unsupported_time::{SystemTime, UNIX_EPOCH};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

impl Instant {
    pub fn now() -> Instant {
        let micros = unsafe { vex_sdk::vexSystemHighResTimeGet() };
        Self(Duration::from_micros(micros))
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0)
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_add(*other)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        Some(Instant(self.0.checked_sub(*other)?))
    }
}
