//! Global `Arc`-based object interning infrastructure.
//!
//! Eventually this should probably be replaced with salsa-based interning.

use std::{
    fmt::{self, Debug},
    hash::{BuildHasherDefault, Hash},
    ops::Deref,
    sync::Arc,
};

use dashmap::{DashMap, SharedValue};
use once_cell::sync::OnceCell;
use rustc_hash::FxHasher;

type InternMap<T> = DashMap<Arc<T>, (), BuildHasherDefault<FxHasher>>;

#[derive(Hash)]
pub struct Interned<T: Internable + ?Sized> {
    arc: Arc<T>,
}

impl<T: Internable> Interned<T> {
    pub fn new(obj: T) -> Self {
        let storage = T::storage().get();
        let shard_idx = storage.determine_map(&obj);
        let shard = &storage.shards()[shard_idx];
        let shard = shard.upgradeable_read();

        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, clone its `Arc` and return it
        //   - if not, box it up, insert it, and return a clone
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.

        // FIXME: avoid double lookup by using raw entry API (once stable, or when hashbrown can be
        // plugged into dashmap)
        if let Some((arc, _)) = shard.get_key_value(&obj) {
            return Self { arc: arc.clone() };
        }

        let arc = Arc::new(obj);
        let arc2 = arc.clone();

        {
            let mut shard = shard.upgrade();
            shard.insert(arc2, SharedValue::new(()));
        }

        Self { arc }
    }
}

impl<T: Internable + ?Sized> Drop for Interned<T> {
    #[inline]
    fn drop(&mut self) {
        // When the last `Ref` is dropped, remove the object from the global map.
        if Arc::strong_count(&self.arc) == 2 {
            // Only `self` and the global map point to the object.

            self.drop_slow();
        }
    }
}

impl<T: Internable + ?Sized> Interned<T> {
    #[cold]
    fn drop_slow(&mut self) {
        let storage = T::storage().get();
        let shard_idx = storage.determine_map(&self.arc);
        let shard = &storage.shards()[shard_idx];
        let mut shard = shard.write();

        // FIXME: avoid double lookup
        let (arc, _) = shard.get_key_value(&self.arc).expect("interned value removed prematurely");

        if Arc::strong_count(arc) != 2 {
            // Another thread has interned another copy
            return;
        }

        shard.remove(&self.arc);

        // Shrink the backing storage if the shard is less than 50% occupied.
        if shard.len() * 2 < shard.capacity() {
            shard.shrink_to_fit();
        }
    }
}

/// Compares interned `Ref`s using pointer equality.
impl<T: Internable + ?Sized> PartialEq for Interned<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.arc, &other.arc)
    }
}

impl<T: Internable + ?Sized> Eq for Interned<T> {}

impl<T: Internable + ?Sized> AsRef<T> for Interned<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.arc
    }
}

impl<T: Internable + ?Sized> Deref for Interned<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl<T: Internable + ?Sized> Clone for Interned<T> {
    fn clone(&self) -> Self {
        Self { arc: self.arc.clone() }
    }
}

impl<T: Debug + Internable + ?Sized> Debug for Interned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*self.arc).fmt(f)
    }
}

pub struct InternStorage<T: ?Sized> {
    map: OnceCell<InternMap<T>>,
}

impl<T: ?Sized> InternStorage<T> {
    pub const fn new() -> Self {
        Self { map: OnceCell::new() }
    }
}

impl<T: Internable + ?Sized> InternStorage<T> {
    fn get(&self) -> &InternMap<T> {
        self.map.get_or_init(DashMap::default)
    }
}

pub trait Internable: Hash + Eq + 'static {
    fn storage() -> &'static InternStorage<Self>;
}

macro_rules! impl_internable {
    ( $($t:ty),+ $(,)? ) => { $(
        impl Internable for $t {
            fn storage() -> &'static InternStorage<Self> {
                static STORAGE: InternStorage<$t> = InternStorage::new();
                &STORAGE
            }
        }
    )+ };
}

impl_internable!(crate::type_ref::TypeRef, crate::type_ref::TraitRef, crate::path::ModPath);
