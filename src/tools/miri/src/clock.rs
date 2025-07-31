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
                // `Duration` does not provide a nice constructor from a `u128` of nanoseconds,
                // so we have to implement this ourselves.
                // It is possible for second to overflow because u64::MAX < (u128::MAX / 1e9).
                // It will be saturated to u64::MAX seconds if the value after division exceeds u64::MAX.
                let seconds = u64::try_from(duration / 1_000_000_000).unwrap_or(u64::MAX);
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
