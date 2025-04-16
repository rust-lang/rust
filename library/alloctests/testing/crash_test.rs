use std::cmp::Ordering;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;

use crate::fmt::Debug; // the `Debug` trait is the only thing we use from `crate::fmt`

/// A blueprint for crash test dummy instances that monitor particular events.
/// Some instances may be configured to panic at some point.
/// Events are `clone`, `drop` or some anonymous `query`.
///
/// Crash test dummies are identified and ordered by an id, so they can be used
/// as keys in a BTreeMap.
#[derive(Debug)]
pub(crate) struct CrashTestDummy {
    pub id: usize,
    cloned: AtomicUsize,
    dropped: AtomicUsize,
    queried: AtomicUsize,
}

impl CrashTestDummy {
    /// Creates a crash test dummy design. The `id` determines order and equality of instances.
    pub(crate) fn new(id: usize) -> CrashTestDummy {
        CrashTestDummy {
            id,
            cloned: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
            queried: AtomicUsize::new(0),
        }
    }

    /// Creates an instance of a crash test dummy that records what events it experiences
    /// and optionally panics.
    pub(crate) fn spawn(&self, panic: Panic) -> Instance<'_> {
        Instance { origin: self, panic }
    }

    /// Returns how many times instances of the dummy have been cloned.
    pub(crate) fn cloned(&self) -> usize {
        self.cloned.load(SeqCst)
    }

    /// Returns how many times instances of the dummy have been dropped.
    pub(crate) fn dropped(&self) -> usize {
        self.dropped.load(SeqCst)
    }

    /// Returns how many times instances of the dummy have had their `query` member invoked.
    pub(crate) fn queried(&self) -> usize {
        self.queried.load(SeqCst)
    }
}

#[derive(Debug)]
pub(crate) struct Instance<'a> {
    origin: &'a CrashTestDummy,
    panic: Panic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Panic {
    Never,
    InClone,
    InDrop,
    InQuery,
}

impl Instance<'_> {
    pub(crate) fn id(&self) -> usize {
        self.origin.id
    }

    /// Some anonymous query, the result of which is already given.
    pub(crate) fn query<R>(&self, result: R) -> R {
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
