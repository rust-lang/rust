use std::cell::Cell;
use std::time::{Duration, Instant as StdInstant, SystemTime};

use crate::MiriMachine;

/// When using a virtual clock, this defines how many nanoseconds we pretend are passing for each
/// basic block.
/// This number is pretty random, but it has been shown to approximately cause
/// some sample programs to run within an order of magnitude of real time on desktop CPUs.
/// (See `tests/pass/shims/time-with-isolation*.rs`.)
const NANOSECONDS_PER_BASIC_BLOCK: u128 = 5000;

/// An instant (a fixed moment in time) in Miri's monotone clock.
#[derive(Clone, Debug)]
pub struct Instant {
    kind: InstantKind,
}

#[derive(Clone, Debug)]
enum InstantKind {
    Host(StdInstant),
    Virtual { nanoseconds: u128 },
}

impl Instant {
    /// Will try to add `duration`, but if that overflows it may add less.
    pub fn add_lossy(&self, duration: Duration) -> Instant {
        match self.kind {
            InstantKind::Host(instant) => {
                // If this overflows, try adding just 1h and assume that will not overflow.
                let i = instant
                    .checked_add(duration)
                    .unwrap_or_else(|| instant.checked_add(Duration::from_secs(3600)).unwrap());
                Instant { kind: InstantKind::Host(i) }
            }
            InstantKind::Virtual { nanoseconds } => {
                let n = nanoseconds.saturating_add(duration.as_nanos());
                Instant { kind: InstantKind::Virtual { nanoseconds: n } }
            }
        }
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        match (&self.kind, earlier.kind) {
            (InstantKind::Host(instant), InstantKind::Host(earlier)) =>
                instant.duration_since(earlier),
            (
                InstantKind::Virtual { nanoseconds },
                InstantKind::Virtual { nanoseconds: earlier },
            ) => {
                let duration = nanoseconds.saturating_sub(earlier);
                Duration::from_nanos_u128(duration)
            }
            _ => panic!("all `Instant` must be of the same kind"),
        }
    }
}

/// A monotone clock used for `Instant` simulation.
#[derive(Debug)]
pub struct MonotonicClock {
    kind: MonotonicClockKind,
}

#[derive(Debug)]
enum MonotonicClockKind {
    Host {
        /// The "epoch" for this machine's monotone clock:
        /// the moment we consider to be time = 0.
        epoch: StdInstant,
    },
    Virtual {
        /// The "current virtual time".
        nanoseconds: Cell<u128>,
    },
}

impl MonotonicClock {
    /// Create a new clock based on the availability of communication with the host.
    pub fn new(communicate: bool) -> Self {
        let kind = if communicate {
            MonotonicClockKind::Host { epoch: StdInstant::now() }
        } else {
            MonotonicClockKind::Virtual { nanoseconds: 0.into() }
        };

        Self { kind }
    }

    /// Let the time pass for a small interval.
    pub fn tick(&self) {
        match &self.kind {
            MonotonicClockKind::Host { .. } => {
                // Time will pass without us doing anything.
            }
            MonotonicClockKind::Virtual { nanoseconds } => {
                nanoseconds.update(|x| x + NANOSECONDS_PER_BASIC_BLOCK);
            }
        }
    }

    /// Sleep for the desired duration.
    pub fn sleep(&self, duration: Duration) {
        match &self.kind {
            MonotonicClockKind::Host { .. } => std::thread::sleep(duration),
            MonotonicClockKind::Virtual { nanoseconds } => {
                // Just pretend that we have slept for some time.
                let nanos: u128 = duration.as_nanos();
                nanoseconds.update(|x| {
                    x.checked_add(nanos)
                        .expect("Miri's virtual clock cannot represent an execution this long")
                });
            }
        }
    }

    /// Return the `epoch` instant (time = 0), to convert between monotone instants and absolute durations.
    pub fn epoch(&self) -> Instant {
        match &self.kind {
            MonotonicClockKind::Host { epoch } => Instant { kind: InstantKind::Host(*epoch) },
            MonotonicClockKind::Virtual { .. } =>
                Instant { kind: InstantKind::Virtual { nanoseconds: 0 } },
        }
    }

    pub fn now(&self) -> Instant {
        match &self.kind {
            MonotonicClockKind::Host { .. } =>
                Instant { kind: InstantKind::Host(StdInstant::now()) },
            MonotonicClockKind::Virtual { nanoseconds } =>
                Instant { kind: InstantKind::Virtual { nanoseconds: nanoseconds.get() } },
        }
    }
}

/// A deadline for some event to occur.
#[derive(Clone, Debug)]
pub enum Deadline {
    Monotonic(Instant),
    RealTime(SystemTime),
}

impl From<Instant> for Deadline {
    fn from(value: Instant) -> Self {
        Deadline::Monotonic(value)
    }
}

impl Deadline {
    /// Will try to add `duration`, but if that overflows it may add less.
    fn add_lossy(&self, duration: Duration) -> Self {
        match self {
            Deadline::Monotonic(i) => Deadline::Monotonic(i.add_lossy(duration)),
            Deadline::RealTime(s) => {
                // If this overflows, try adding just 1h and assume that will not overflow.
                Deadline::RealTime(
                    s.checked_add(duration)
                        .unwrap_or_else(|| s.checked_add(Duration::from_secs(3600)).unwrap()),
                )
            }
        }
    }
}

/// The clock to use for the timeout you are asking for.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TimeoutClock {
    /// The timeout is measured using the monotone clock.
    Monotonic,
    /// The timeout is measured using the host's system clock.
    RealTime,
}

/// Whether the timeout is relative or absolute.
#[derive(Debug, Copy, Clone)]
pub enum TimeoutStyle {
    /// The given duration is interpreted relative to "now" for the selected clock.
    Relative,
    /// The given duration is interpreted as an "absolute" time, i.e., relative to the epoch of the
    /// selected clock.
    Absolute,
}

impl MiriMachine<'_> {
    /// Computes the deadline for a given timeout configuration and duration.
    pub fn timeout(
        &self,
        clock: TimeoutClock,
        style: TimeoutStyle,
        duration: Duration,
    ) -> Deadline {
        // First let's figure out what "zero" means for the given clock and style.
        let zero = match clock {
            TimeoutClock::RealTime => {
                assert!(self.communicate(), "cannot have `RealTime` timeout with isolation");
                Deadline::RealTime(match style {
                    TimeoutStyle::Absolute => SystemTime::UNIX_EPOCH,
                    TimeoutStyle::Relative => SystemTime::now(),
                })
            }
            TimeoutClock::Monotonic =>
                Deadline::Monotonic(match style {
                    TimeoutStyle::Absolute => self.monotonic_clock.epoch(),
                    TimeoutStyle::Relative => self.monotonic_clock.now(),
                }),
        };
        // Then add the given duration relative to that "zero".
        zero.add_lossy(duration)
    }
}
