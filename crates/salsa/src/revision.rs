//!
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};

/// Value of the initial revision, as a u32. We don't use 0
/// because we want to use a `NonZeroU32`.
const START: u32 = 1;

/// A unique identifier for the current version of the database; each
/// time an input is changed, the revision number is incremented.
/// `Revision` is used internally to track which values may need to be
/// recomputed, but is not something you should have to interact with
/// directly as a user of salsa.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Revision {
    generation: NonZeroU32,
}

impl Revision {
    pub(crate) fn start() -> Self {
        Self::from(START)
    }

    pub(crate) fn from(g: u32) -> Self {
        Self { generation: NonZeroU32::new(g).unwrap() }
    }

    pub(crate) fn next(self) -> Revision {
        Self::from(self.generation.get() + 1)
    }

    fn as_u32(self) -> u32 {
        self.generation.get()
    }
}

impl std::fmt::Debug for Revision {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "R{}", self.generation)
    }
}

#[derive(Debug)]
pub(crate) struct AtomicRevision {
    data: AtomicU32,
}

impl AtomicRevision {
    pub(crate) fn start() -> Self {
        Self { data: AtomicU32::new(START) }
    }

    pub(crate) fn load(&self) -> Revision {
        Revision::from(self.data.load(Ordering::SeqCst))
    }

    pub(crate) fn store(&self, r: Revision) {
        self.data.store(r.as_u32(), Ordering::SeqCst);
    }

    /// Increment by 1, returning previous value.
    pub(crate) fn fetch_then_increment(&self) -> Revision {
        let v = self.data.fetch_add(1, Ordering::SeqCst);
        assert!(v != u32::max_value(), "revision overflow");
        Revision::from(v)
    }
}
