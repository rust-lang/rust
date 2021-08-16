use crate::fmt::Debug;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

/// A blueprint for crash test dummy instances that monitor particular events.
/// Some instances may be configured to panic at some point.
/// Events are `clone`, `drop` or some anonymous `query`.
///
/// Crash test dummies are identified and ordered by an id, so they can be used
/// as keys in a BTreeMap. The implementation intentionally uses does not rely
/// on anything defined in the crate, apart from the `Debug` trait.
#[derive(Debug)]
pub struct CrashTestDummy {
    pub id: usize,
    cloned: AtomicUsize,
    dropped: AtomicUsize,
    queried: AtomicUsize,
}

impl CrashTestDummy {
    /// Creates a crash test dummy design. The `id` determines order and equality of instances.
    pub fn new(id: usize) -> CrashTestDummy {
        CrashTestDummy {
            id,
            cloned: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
            queried: AtomicUsize::new(0),
        }
    }

    /// Creates an instance of a crash test dummy that records what events it experiences
    /// and optionally panics.
    pub fn spawn(&self, panic: Panic) -> Instance<'_> {
        Instance { origin: self, panic }
    }

    /// Returns how many times instances of the dummy have been cloned.
    pub fn cloned(&self) -> usize {
        self.cloned.load(SeqCst)
    }

    /// Returns how many times instances of the dummy have been dropped.
    pub fn dropped(&self) -> usize {
        self.dropped.load(SeqCst)
    }

    /// Returns how many times instances of the dummy have had their `query` member invoked.
    pub fn queried(&self) -> usize {
        self.queried.load(SeqCst)
    }
}

#[derive(Debug)]
pub struct Instance<'a> {
    origin: &'a CrashTestDummy,
    panic: Panic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Panic {
    Never,
    InClone,
    InDrop,
    InQuery,
}

impl Instance<'_> {
    pub fn id(&self) -> usize {
        self.origin.id
    }

    /// Some anonymous query, the result of which is already given.
    pub fn query<R>(&self, result: R) -> R {
        self.origin.queried.fetch_add(1, SeqCst);
        if self.panic == Panic::InQuery {
            panic!("panic in `query`");
        }
        result
    }
}

impl Clone for Instance<'_> {
    fn clone(&self) -> Self {
        self.origin.cloned.fetch_add(1, SeqCst);
        if self.panic == Panic::InClone {
            panic!("panic in `clone`");
        }
        Self { origin: self.origin, panic: Panic::Never }
    }
}

impl Drop for Instance<'_> {
    fn drop(&mut self) {
        self.origin.dropped.fetch_add(1, SeqCst);
        if self.panic == Panic::InDrop {
            panic!("panic in `drop`");
        }
    }
}

impl PartialOrd for Instance<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id().partial_cmp(&other.id())
    }
}

impl Ord for Instance<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id().cmp(&other.id())
    }
}

impl PartialEq for Instance<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id().eq(&other.id())
    }
}

impl Eq for Instance<'_> {}
