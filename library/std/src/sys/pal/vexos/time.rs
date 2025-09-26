use crate::time::Duration;

#[expect(dead_code)]
#[path = "../unsupported/time.rs"]
mod unsupported_time;
pub use unsupported_time::{SystemTime, UNIX_EPOCH};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant {
    micros: u64,
}

impl Instant {
    pub fn now() -> Instant {
        let micros = unsafe { vex_sdk::vexSystemHighResTimeGet() };
        Self { micros }
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        let micros = self.micros.checked_sub(other.micros)?;
        Some(Duration::from_micros(micros))
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        let to_add = other.as_micros().try_into().ok()?;
        let micros = self.micros.checked_add(to_add)?;
        Some(Instant { micros })
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        let to_sub = other.as_micros().try_into().ok()?;
        let micros = self.micros.checked_sub(to_sub)?;
        Some(Instant { micros })
    }
}
