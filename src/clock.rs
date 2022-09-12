use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant as StdInstant};

/// When using a virtual clock, this defines how many nanoseconds we pretend are passing for each
/// basic block.
const NANOSECONDS_PER_BASIC_BLOCK: u64 = 10;

#[derive(Debug)]
pub struct Instant {
    kind: InstantKind,
}

#[derive(Debug)]
enum InstantKind {
    Host(StdInstant),
    Virtual { nanoseconds: u64 },
}

impl Instant {
    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        match self.kind {
            InstantKind::Host(instant) =>
                instant.checked_add(duration).map(|i| Instant { kind: InstantKind::Host(i) }),
            InstantKind::Virtual { nanoseconds } =>
                u128::from(nanoseconds)
                    .checked_add(duration.as_nanos())
                    .and_then(|n| u64::try_from(n).ok())
                    .map(|nanoseconds| Instant { kind: InstantKind::Virtual { nanoseconds } }),
        }
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        match (&self.kind, earlier.kind) {
            (InstantKind::Host(instant), InstantKind::Host(earlier)) =>
                instant.duration_since(earlier),
            (
                InstantKind::Virtual { nanoseconds },
                InstantKind::Virtual { nanoseconds: earlier },
            ) => Duration::from_nanos(nanoseconds.saturating_sub(earlier)),
            _ => panic!("all `Instant` must be of the same kind"),
        }
    }
}

/// A monotone clock used for `Instant` simulation.
#[derive(Debug)]
pub struct Clock {
    kind: ClockKind,
}

#[derive(Debug)]
enum ClockKind {
    Host {
        /// The "time anchor" for this machine's monotone clock.
        time_anchor: StdInstant,
    },
    Virtual {
        /// The "current virtual time".
        nanoseconds: AtomicU64,
    },
}

impl Clock {
    /// Create a new clock based on the availability of communication with the host.
    pub fn new(communicate: bool) -> Self {
        let kind = if communicate {
            ClockKind::Host { time_anchor: StdInstant::now() }
        } else {
            ClockKind::Virtual { nanoseconds: 0.into() }
        };

        Self { kind }
    }

    /// Let the time pass for a small interval.
    pub fn tick(&self) {
        match &self.kind {
            ClockKind::Host { .. } => {
                // Time will pass without us doing anything.
            }
            ClockKind::Virtual { nanoseconds } => {
                nanoseconds.fetch_add(NANOSECONDS_PER_BASIC_BLOCK, Ordering::SeqCst);
            }
        }
    }

    /// Sleep for the desired duration.
    pub fn sleep(&self, duration: Duration) {
        match &self.kind {
            ClockKind::Host { .. } => std::thread::sleep(duration),
            ClockKind::Virtual { nanoseconds } => {
                // Just pretend that we have slept for some time.
                nanoseconds.fetch_add(duration.as_nanos().try_into().unwrap(), Ordering::SeqCst);
            }
        }
    }

    /// Return the `anchor` instant, to convert between monotone instants and durations relative to the anchor.
    pub fn anchor(&self) -> Instant {
        match &self.kind {
            ClockKind::Host { time_anchor } => Instant { kind: InstantKind::Host(*time_anchor) },
            ClockKind::Virtual { .. } => Instant { kind: InstantKind::Virtual { nanoseconds: 0 } },
        }
    }

    pub fn now(&self) -> Instant {
        match &self.kind {
            ClockKind::Host { .. } => Instant { kind: InstantKind::Host(StdInstant::now()) },
            ClockKind::Virtual { nanoseconds } =>
                Instant {
                    kind: InstantKind::Virtual { nanoseconds: nanoseconds.load(Ordering::SeqCst) },
                },
        }
    }
}
