//! Server-side handles and storage for per-handle data.

use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::num::NonZeroU32;
use std::ops::{Index, IndexMut};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) type Handle = NonZeroU32;

pub(super) struct OwnedStore<T: 'static> {
    counter: &'static AtomicUsize,
    data: BTreeMap<Handle, T>,
}

impl<T> OwnedStore<T> {
    pub(super) fn new(counter: &'static AtomicUsize) -> Self {
        // Ensure the handle counter isn't 0, which would panic later,
        // when `NonZeroU32::new` (aka `Handle::new`) is called in `alloc`.
        assert_ne!(counter.load(Ordering::SeqCst), 0);

        OwnedStore { counter, data: BTreeMap::new() }
    }
}

impl<T> OwnedStore<T> {
    pub(super) fn alloc(&mut self, x: T) -> Handle {
        let counter = self.counter.fetch_add(1, Ordering::SeqCst);
        let handle = Handle::new(counter as u32).expect("`proc_macro` handle counter overflowed");
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

pub(super) trait FromKey<Q: ?Sized> {
    fn from_key(key: &Q) -> Self;
}

impl<T: Clone> FromKey<T> for T {
    fn from_key(key: &T) -> T {
        key.clone()
    }
}

impl<T: ?Sized> FromKey<T> for Rc<T>
where
    Rc<T>: for<'a> From<&'a T>,
{
    fn from_key(key: &T) -> Rc<T> {
        key.into()
    }
}

pub(super) struct InternedStore<T: 'static> {
    owned: OwnedStore<T>,
    interner: HashMap<T, Handle>,
}

impl<T: Clone + Eq + Hash> InternedStore<T> {
    pub(super) fn new(counter: &'static AtomicUsize) -> Self {
        InternedStore { owned: OwnedStore::new(counter), interner: HashMap::new() }
    }

    pub(super) fn alloc<'a, Q: ?Sized>(&mut self, x: &'a Q) -> Handle
    where
        T: Borrow<Q> + FromKey<Q>,
        Q: Hash + Eq,
    {
        let owned = &mut self.owned;
        *self
            .interner
            .raw_entry_mut()
            .from_key(x)
            .or_insert_with(|| {
                let own = T::from_key(x);
                (own.clone(), owned.alloc(own))
            })
            .1
    }

    pub(super) fn copy(&mut self, h: Handle) -> T {
        self.owned[h].clone()
    }
}

impl<T> Index<Handle> for InternedStore<T> {
    type Output = T;
    fn index(&self, h: Handle) -> &T {
        self.owned.index(h)
    }
}
