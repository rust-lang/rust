//! Server-side handles and storage for per-handle data.

use std::collections::BTreeMap;
use std::hash::Hash;
use std::num::NonZero;
use std::sync::atomic::{AtomicU32, Ordering};

use super::fxhash::FxHashMap;

pub(super) type Handle = NonZero<u32>;

/// A store that associates values of type `T` with numeric handles. A value can
/// be looked up using its handle. Avoids storing any value more than once.
pub(super) struct InternedStore<T: 'static> {
    counter: &'static AtomicU32,
    data: BTreeMap<Handle, T>,
    interner: FxHashMap<T, Handle>,
}

impl<T: Copy + Eq + Hash> InternedStore<T> {
    pub(super) fn new(counter: &'static AtomicU32) -> Self {
        // Ensure the handle counter isn't 0, which would panic later,
        // when `NonZero::new` (aka `Handle::new`) is called in `alloc`.
        assert_ne!(counter.load(Ordering::Relaxed), 0);

        InternedStore { counter, data: BTreeMap::new(), interner: FxHashMap::default() }
    }

    pub(super) fn alloc(&mut self, x: T) -> Handle {
        *self.interner.entry(x).or_insert_with(|| {
            let counter = self.counter.fetch_add(1, Ordering::Relaxed);
            let handle = Handle::new(counter).expect("`proc_macro` handle counter overflowed");
            assert!(self.data.insert(handle, x).is_none());
            handle
        })
    }

    pub(super) fn copy(&mut self, h: Handle) -> T {
        *self.data.get(&h).expect("use-after-free in `proc_macro` handle")
    }
}
