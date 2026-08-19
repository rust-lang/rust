use crate::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(Duration);

impl Instant {
    pub fn now() -> Instant {
        let micros = unsafe { vex_sdk::vexSystemHighResTimeGet() };
        Self(Duration::from_micros(micros))
    }

    pub fn from_duration(duration: Duration) -> Instant {
        Instant(duration)
    }

    pub fn into_duration(self) -> Duration {
        self.0
    }
}
