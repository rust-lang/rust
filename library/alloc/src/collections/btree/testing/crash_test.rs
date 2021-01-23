use crate::fmt::Debug;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};

/// A recording of particular events happening in a test scenario that involves
/// crash test dummies, some of which may be configured to panic at some point.
/// Events are `clone`, `drop` or some anonymous `query`.
///
/// Crash test dummies are identified and ordered by an id, so they can be used
/// as keys in a BTreeMap. The implementation intentionally uses only primitives.
#[derive(Debug)]
pub struct CrashTest {
    bits_per_id: usize,
    cloned: AtomicU64,
    dropped: AtomicU64,
    queried: AtomicU64,
}

impl CrashTest {
    /// Sets up the recording of a test scenario that counts how many events
    /// dummies experience, per event type and per dummy id. The range of dummy
    /// ids is limited to 16 and the number of times all dummies with the same
    /// dummy id experience the same event is also limited to 16.
    pub fn new() -> Self {
        Self {
            bits_per_id: 4,
            cloned: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            queried: AtomicU64::new(0),
        }
    }

    /// Sets up the recording of a test scenario that counts how many events
    /// dummies experience, per event type. There is virtually no limit on the
    /// range of dummy ids or on the number of events happening.
    pub fn new_totaling() -> Self {
        Self {
            bits_per_id: 0,
            cloned: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            queried: AtomicU64::new(0),
        }
    }

    /// Creates a crash test dummy that records what events it experiences
    /// and optionally panics.
    pub fn dummy(&self, id: usize, panic: Panic) -> Dummy<'_> {
        assert!(self.is_supported_id(id));
        Dummy { id, panic, context: self }
    }

    /// Returns how many times a dummy has been cloned.  If set up by `new`,
    /// this is a hexadecimal composition of the count for each dummy id.
    /// If set up by `new_totaling`, this is the sum over all dummies.
    pub fn cloned(&self) -> u64 {
        self.cloned.load(SeqCst)
    }

    /// Returns how many times a dummy has been dropped. If set up by `new`,
    /// this is a hexadecimal composition of the count for each dummy id.
    /// If set up by `new_totaling`, this is the sum over all dummies.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(SeqCst)
    }

    /// Returns how many times a dummy has been queried. If set up by `new`,
    /// this is a hexadecimal composition of the count for each dummy id.
    /// If set up by `new_totaling`, this is the sum over all dummies.
    pub fn queried(&self) -> u64 {
        self.queried.load(SeqCst)
    }

    /// Whether there is room for the id as a counter in our registers.
    fn is_supported_id(&self, id: usize) -> bool {
        (id + 1) * self.bits_per_id <= 64
    }

    /// Bit position of the counter in our registers.
    fn bit(&self, id: usize) -> u64 {
        1 << (id * self.bits_per_id)
    }
}

#[derive(Debug)]
pub struct Dummy<'a> {
    pub id: usize,
    context: &'a CrashTest,
    panic: Panic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Panic {
    Never,
    InClone,
    InDrop,
    InQuery,
}

impl Dummy<'_> {
    /// Some anonymous query, the result of which is already given.
    pub fn query<R>(&self, result: R) -> R {
        self.context.queried.fetch_add(self.context.bit(self.id), SeqCst);
        if self.panic == Panic::InQuery {
            panic!("panic in `query`");
        }
        result
    }
}

impl Clone for Dummy<'_> {
    fn clone(&self) -> Self {
        self.context.cloned.fetch_add(self.context.bit(self.id), SeqCst);
        if self.panic == Panic::InClone {
            panic!("panic in `clone`");
        }
        Self { id: self.id, context: self.context, panic: Panic::Never }
    }
}

impl Drop for Dummy<'_> {
    fn drop(&mut self) {
        self.context.dropped.fetch_add(self.context.bit(self.id), SeqCst);
        if self.panic == Panic::InDrop {
            panic!("panic in `drop`");
        }
    }
}

impl PartialOrd for Dummy<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for Dummy<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialEq for Dummy<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl Eq for Dummy<'_> {}
