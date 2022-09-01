use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant as StdInstant};

use rustc_data_structures::sync::Ordering;

use crate::*;

/// When using a virtual clock, this defines how many nanoseconds do we pretend
/// are passing for each basic block.
const NANOSECOND_PER_BASIC_BLOCK: u64 = 10;

#[derive(Debug)]
pub struct Instant {
    kind: InstantKind,
}

#[derive(Debug)]
enum InstantKind {
    Host(StdInstant),
    Virtual { nanoseconds: u64 },
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

    /// Get the current time relative to this clock.
    pub fn get(&self) -> Duration {
        match &self.kind {
            ClockKind::Host { time_anchor } =>
                StdInstant::now().saturating_duration_since(*time_anchor),
            ClockKind::Virtual { nanoseconds } =>
                Duration::from_nanos(nanoseconds.load(Ordering::Relaxed)),
        }
    }

    /// Let the time pass for a small interval.
    pub fn tick(&self) {
        match &self.kind {
            ClockKind::Host { .. } => {
                // Time will pass without us doing anything.
            }
            ClockKind::Virtual { nanoseconds } => {
                nanoseconds.fetch_add(NANOSECOND_PER_BASIC_BLOCK, Ordering::Relaxed);
            }
        }
    }

    /// Sleep for the desired duration.
    pub fn sleep(&self, duration: Duration) {
        match &self.kind {
            ClockKind::Host { .. } => std::thread::sleep(duration),
            ClockKind::Virtual { nanoseconds } => {
                // Just pretend that we have slept for some time.
                nanoseconds.fetch_add(duration.as_nanos().try_into().unwrap(), Ordering::Relaxed);
            }
        }
    }

    /// Compute `now + duration` relative to this clock.
    pub fn get_time_relative(&self, duration: Duration) -> Option<Instant> {
        match &self.kind {
            ClockKind::Host { .. } =>
                StdInstant::now()
                    .checked_add(duration)
                    .map(|instant| Instant { kind: InstantKind::Host(instant) }),
            ClockKind::Virtual { nanoseconds } =>
                nanoseconds
                    .load(Ordering::Relaxed)
                    .checked_add(duration.as_nanos().try_into().unwrap())
                    .map(|nanoseconds| Instant { kind: InstantKind::Virtual { nanoseconds } }),
        }
    }

    /// Compute `start + duration` relative to this clock where `start` is the instant of time when
    /// this clock was created.
    pub fn get_time_absolute(&self, duration: Duration) -> Option<Instant> {
        match &self.kind {
            ClockKind::Host { time_anchor } =>
                time_anchor
                    .checked_add(duration)
                    .map(|instant| Instant { kind: InstantKind::Host(instant) }),
            ClockKind::Virtual { .. } =>
                Some(Instant {
                    kind: InstantKind::Virtual {
                        nanoseconds: duration.as_nanos().try_into().unwrap(),
                    },
                }),
        }
    }

    /// Returns the duration until the given instant.
    pub fn duration_until(&self, instant: &Instant) -> Duration {
        match (&instant.kind, &self.kind) {
            (InstantKind::Host(instant), ClockKind::Host { .. }) =>
                instant.saturating_duration_since(StdInstant::now()),
            (
                InstantKind::Virtual { nanoseconds },
                ClockKind::Virtual { nanoseconds: current_ns },
            ) =>
                Duration::from_nanos(nanoseconds.saturating_sub(current_ns.load(Ordering::Relaxed))),
            _ => panic!(),
        }
    }
}
