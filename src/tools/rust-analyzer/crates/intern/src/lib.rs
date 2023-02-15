//! Global `Arc`-based object interning infrastructure.
//!
//! Eventually this should probably be replaced with salsa-based interning.

use std::{
    fmt::{self, Debug, Display},
    hash::{BuildHasherDefault, Hash, Hasher},
    ops::Deref,
    sync::Arc,
};

use dashmap::{DashMap, SharedValue};
use hashbrown::HashMap;
use once_cell::sync::OnceCell;
use rustc_hash::FxHasher;

type InternMap<T> = DashMap<Arc<T>, (), BuildHasherDefault<FxHasher>>;
type Guard<T> = dashmap::RwLockWriteGuard<
    'static,
    HashMap<Arc<T>, SharedValue<()>, BuildHasherDefault<FxHasher>>,
>;

pub struct Interned<T: Internable + ?Sized> {
    arc: Arc<T>,
}

impl<T: Internable> Interned<T> {
    pub fn new(obj: T) -> Self {
        match Interned::lookup(&obj) {
            Ok(this) => this,
            Err(shard) => {
                let arc = Arc::new(obj);
                Self::alloc(arc, shard)
            }
        }
    }
}

impl<T: Internable + ?Sized> Interned<T> {
    fn lookup(obj: &T) -> Result<Self, Guard<T>> {
        let storage = T::storage().get();
        let shard_idx = storage.determine_map(obj);
        let shard = &storage.shards()[shard_idx];
        let shard = shard.write();

        // Atomically,
        // - check if `obj` is already in the map
        //   - if so, clone its `Arc` and return it
        //   - if not, box it up, insert it, and return a clone
        // This needs to be atomic (locking the shard) to avoid races with other thread, which could
        // insert the same object between us looking it up and inserting it.

        // FIXME: avoid double lookup/hashing by using raw entry API (once stable, or when
        // hashbrown can be plugged into dashmap)
        match shard.get_key_value(obj) {
            Some((arc, _)) => Ok(Self { arc: arc.clone() }),
            None => Err(shard),
        }
    }

    fn alloc(arc: Arc<T>, mut shard: Guard<T>) -> Self {
        let arc2 = arc.clone();

        shard.insert(arc2, SharedValue::new(()));

        Self { arc }
    }
}

impl Interned<str> {
    pub fn new_str(s: &str) -> Self {
        match Interned::lookup(s) {
            Ok(this) => this,
            Err(shard) => {
                let arc = Arc::<str>::from(s);
                Self::alloc(arc, shard)
            }
        }
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
impl<T: Internable> PartialEq for Interned<T> {
    // NOTE: No `?Sized` because `ptr_eq` doesn't work right with trait objects.

    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.arc, &other.arc)
    }
}

impl<T: Internable> Eq for Interned<T> {}

impl PartialEq for Interned<str> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.arc, &other.arc)
    }
}

impl Eq for Interned<str> {}

impl<T: Internable + ?Sized> Hash for Interned<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // NOTE: Cast disposes vtable pointer / slice/str length.
        state.write_usize(Arc::as_ptr(&self.arc) as *const () as usize)
    }
}

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

impl<T: Display + Internable + ?Sized> Display for Interned<T> {
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

/// Implements `Internable` for a given list of types, making them usable with `Interned`.
#[macro_export]
#[doc(hidden)]
macro_rules! _impl_internable {
    ( $($t:path),+ $(,)? ) => { $(
        impl $crate::Internable for $t {
            fn storage() -> &'static $crate::InternStorage<Self> {
                static STORAGE: $crate::InternStorage<$t> = $crate::InternStorage::new();
                &STORAGE
            }
        }
    )+ };
}

pub use crate::_impl_internable as impl_internable;

impl_internable!(str,);
