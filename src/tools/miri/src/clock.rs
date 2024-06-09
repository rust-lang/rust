use std::cell::Cell;
use std::time::{Duration, Instant as StdInstant};

/// When using a virtual clock, this defines how many nanoseconds we pretend are passing for each
/// basic block.
/// This number is pretty random, but it has been shown to approximately cause
/// some sample programs to run within an order of magnitude of real time on desktop CPUs.
/// (See `tests/pass/shims/time-with-isolation*.rs`.)
const NANOSECONDS_PER_BASIC_BLOCK: u128 = 5000;

#[derive(Debug)]
pub struct Instant {
    kind: InstantKind,
}

#[derive(Debug)]
enum InstantKind {
    Host(StdInstant),
    Virtual { nanoseconds: u128 },
}

impl Instant {
    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        match self.kind {
            InstantKind::Host(instant) =>
                instant.checked_add(duration).map(|i| Instant { kind: InstantKind::Host(i) }),
            InstantKind::Virtual { nanoseconds } =>
                nanoseconds
                    .checked_add(duration.as_nanos())
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
            ) => {
                let duration = nanoseconds.saturating_sub(earlier);
                // `Duration` does not provide a nice constructor from a `u128` of nanoseconds,
                // so we have to implement this ourselves.
                // It is possible for second to overflow because u64::MAX < (u128::MAX / 1e9).
                let seconds = u64::try_from(duration.saturating_div(1_000_000_000)).unwrap();
                // It is impossible for nanosecond to overflow because u32::MAX > 1e9.
                let nanosecond = u32::try_from(duration.wrapping_rem(1_000_000_000)).unwrap();
                Duration::new(seconds, nanosecond)
            }
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
        nanoseconds: Cell<u128>,
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
                nanoseconds.update(|x| x + NANOSECONDS_PER_BASIC_BLOCK);
            }
        }
    }

    /// Sleep for the desired duration.
    pub fn sleep(&self, duration: Duration) {
        match &self.kind {
            ClockKind::Host { .. } => std::thread::sleep(duration),
            ClockKind::Virtual { nanoseconds } => {
                // Just pretend that we have slept for some time.
                let nanos: u128 = duration.as_nanos();
                nanoseconds.update(|x| x.checked_add(nanos).expect("Miri's virtual clock cannot represent an execution this long"));
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
                Instant { kind: InstantKind::Virtual { nanoseconds: nanoseconds.get() } },
        }
    }
}
