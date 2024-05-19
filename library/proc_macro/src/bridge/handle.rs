//! Server-side handles and storage for per-handle data.

use std::collections::BTreeMap;
use std::hash::Hash;
use std::num::NonZero;
use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicU32, Ordering};

use super::fxhash::FxHashMap;

pub(super) type Handle = NonZero<u32>;

/// A store that associates values of type `T` with numeric handles. A value can
/// be looked up using its handle.
pub(super) struct OwnedStore<T: 'static> {
    counter: &'static AtomicU32,
    data: BTreeMap<Handle, T>,
}

impl<T> OwnedStore<T> {
    pub(super) fn new(counter: &'static AtomicU32) -> Self {
        // Ensure the handle counter isn't 0, which would panic later,
        // when `NonZero::new` (aka `Handle::new`) is called in `alloc`.
        assert_ne!(counter.load(Ordering::Relaxed), 0);

        OwnedStore { counter, data: BTreeMap::new() }
    }
}

impl<T> OwnedStore<T> {
    pub(super) fn alloc(&mut self, x: T) -> Handle {
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        let handle = Handle::new(counter).expect("`proc_macro` handle counter overflowed");
        assert!(self.data.insert(handle, x).is_none());
        handle
    }

    pub(super) fn take(&mut self, h: Handle) -> T {
        self.data.remove(&h).expect("use-after-free in `proc_macro` handle")
    }
}

impl<T> Index<Handle> for OwnedStore<T> {
    type Output = T;
    fn index(&self, h: Handle) -> &T {
        self.data.get(&h).expect("use-after-free in `proc_macro` handle")
    }
}

impl<T> IndexMut<Handle> for OwnedStore<T> {
    fn index_mut(&mut self, h: Handle) -> &mut T {
        self.data.get_mut(&h).expect("use-after-free in `proc_macro` handle")
    }
}

/// Like `OwnedStore`, but avoids storing any value more than once.
pub(super) struct InternedStore<T: 'static> {
    owned: OwnedStore<T>,
    interner: FxHashMap<T, Handle>,
}

impl<T: Copy + Eq + Hash> InternedStore<T> {
    pub(super) fn new(counter: &'static AtomicU32) -> Self {
        InternedStore { owned: OwnedStore::new(counter), interner: FxHashMap::default() }
    }

    pub(super) fn alloc(&mut self, x: T) -> Handle {
        let owned = &mut self.owned;
        *self.interner.entry(x).or_insert_with(|| owned.alloc(x))
    }

    pub(super) fn copy(&mut self, h: Handle) -> T {
        self.owned[h]
    }
}
