use crate::fx::{FxHashMap, FxHasher};
use crate::sync::{CacheAligned, Lock, LockGuard};
use std::borrow::Borrow;
use std::collections::hash_map::RawEntryMut;
use std::hash::{Hash, Hasher};

const SHARD_BITS: usize = 0;

pub const SHARDS: usize = 1 << SHARD_BITS;

/// An array of cache-line aligned inner locked structures with convenience methods.
pub struct Sharded<T> {
    shard: Lock<T>,
}

impl<T: Default> Default for Sharded<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default)
    }
}

impl<T: Default> Sharded<T> {
    #[inline]
    pub fn new(mut value: impl FnMut() -> T) -> Self {
        Sharded { shard: Lock::new(value()) }
    }

    /// The shard is selected by hashing `val` with `FxHasher`.
    #[inline]
    pub fn with_get_shard_by_value<K: Hash + ?Sized, F: FnOnce(&mut T) -> R, R>(
        &self,
        _val: &K,
        f: F,
    ) -> R {
        self.shard.with_lock(f)
    }

    #[inline]
    pub fn with_get_shard_by_hash<F: FnOnce(&mut T) -> R, R>(&self, _hash: u64, f: F) -> R {
        self.shard.with_lock(f)
    }

    #[inline]
    pub fn get_shard_by_value<K: Hash + ?Sized>(&self, _val: &K) -> &Lock<T> {
        &self.shard
    }

    #[inline]
    pub fn get_shard_by_hash(&self, _hash: u64) -> &Lock<T> {
        &self.shard
    }

    pub fn lock_shards(&self) -> Vec<LockGuard<'_, T>> {
        vec![self.shard.lock()]
    }

    pub fn try_lock_shards(&self) -> Option<Vec<LockGuard<'_, T>>> {
        Some(vec![self.shard.try_lock()?])
    }
}

pub type ShardedHashMap<K, V> = Sharded<FxHashMap<K, V>>;

impl<K: Eq, V> ShardedHashMap<K, V> {
    pub fn len(&self) -> usize {
        self.lock_shards().iter().map(|shard| shard.len()).sum()
    }
}

impl<K: Eq + Hash + Copy> ShardedHashMap<K, ()> {
    #[inline]
    pub fn intern_ref<Q: ?Sized>(&self, value: &Q, make: impl FnOnce() -> K) -> K
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash(value);
        self.with_get_shard_by_hash(hash, |shard| {
            let entry = shard.raw_entry_mut().from_key_hashed_nocheck(hash, value);

            match entry {
                RawEntryMut::Occupied(e) => *e.key(),
                RawEntryMut::Vacant(e) => {
                    let v = make();
                    e.insert_hashed_nocheck(hash, v, ());
                    v
                }
            }
        })
    }

    #[inline]
    pub fn intern<Q>(&self, value: Q, make: impl FnOnce(Q) -> K) -> K
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash(&value);
        self.with_get_shard_by_hash(hash, |shard| {
            let entry = shard.raw_entry_mut().from_key_hashed_nocheck(hash, &value);

            match entry {
                RawEntryMut::Occupied(e) => *e.key(),
                RawEntryMut::Vacant(e) => {
                    let v = make(value);
                    e.insert_hashed_nocheck(hash, v, ());
                    v
                }
            }
        })
    }
}

pub trait IntoPointer {
    /// Returns a pointer which outlives `self`.
    fn into_pointer(&self) -> *const ();
}

impl<K: Eq + Hash + Copy + IntoPointer> ShardedHashMap<K, ()> {
    pub fn contains_pointer_to<T: Hash + IntoPointer>(&self, value: &T) -> bool {
        let hash = make_hash(&value);

        self.with_get_shard_by_hash(hash, |shard| {
            let value = value.into_pointer();
            shard.raw_entry().from_hash(hash, |entry| entry.into_pointer() == value).is_some()
        })
    }
}

#[inline]
pub fn make_hash<K: Hash + ?Sized>(val: &K) -> u64 {
    let mut state = FxHasher::default();
    val.hash(&mut state);
    state.finish()
}

/*
/// Get a shard with a pre-computed hash value. If `get_shard_by_value` is
/// ever used in combination with `get_shard_by_hash` on a single `Sharded`
/// instance, then `hash` must be computed with `FxHasher`. Otherwise,
/// `hash` can be computed with any hasher, so long as that hasher is used
/// consistently for each `Sharded` instance.
#[inline]
#[allow(clippy::modulo_one)]
fn get_shard_index_by_hash(hash: u64) -> usize {
    let hash_len = mem::size_of::<usize>();
    // Ignore the top 7 bits as hashbrown uses these and get the next SHARD_BITS highest bits.
    // hashbrown also uses the lowest bits, so we can't use those
    let bits = (hash >> (hash_len * 8 - 7 - SHARD_BITS)) as usize;
    bits % SHARDS
}
*/
