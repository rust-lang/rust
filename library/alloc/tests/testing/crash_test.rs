use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;

/// A blueprint for crash test dummy instances that monitor drops.
/// Some instances may be configured to panic at some point.
///
/// Crash test dummies are identified and ordered by an id, so they can be used
/// as keys in a BTreeMap.
#[derive(Debug)]
pub struct CrashTestDummy {
    pub id: usize,
    dropped: AtomicUsize,
}

impl CrashTestDummy {
    /// Creates a crash test dummy design. The `id` determines order and equality of instances.
    pub fn new(id: usize) -> CrashTestDummy {
        CrashTestDummy { id, dropped: AtomicUsize::new(0) }
    }

    /// Creates an instance of a crash test dummy that records what events it experiences
    /// and optionally panics.
    pub fn spawn(&self, panic: Panic) -> Instance<'_> {
        Instance { origin: self, panic }
    }

    /// Returns how many times instances of the dummy have been dropped.
    pub fn dropped(&self) -> usize {
        self.dropped.load(SeqCst)
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
    InDrop,
}

impl Instance<'_> {
    pub fn id(&self) -> usize {
        self.origin.id
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
