use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::{iter, mem};

use either::Either;
use hashbrown::hash_table::{Entry, HashTable};

use crate::fx::FxHasher;
use crate::sync::{CacheAligned, Lock, LockGuard, Mode, is_dyn_thread_safe};

// 32 shards is sufficient to reduce contention on an 8-core Ryzen 7 1700,
// but this should be tested on higher core count CPUs. How the `Sharded` type gets used
// may also affect the ideal number of shards.
const SHARD_BITS: usize = 5;

const SHARDS: usize = 1 << SHARD_BITS;

/// An array of cache-line aligned inner locked structures with convenience methods.
/// A single field is used when the compiler uses only one thread.
pub enum Sharded<T> {
    Single(Lock<T>),
    Shards(Box<[CacheAligned<Lock<T>>; SHARDS]>),
}

impl<T: Default> Default for Sharded<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default)
    }
}

impl<T> Sharded<T> {
    #[inline]
    pub fn new(mut value: impl FnMut() -> T) -> Self {
        if is_dyn_thread_safe() {
            return Sharded::Shards(Box::new(
                [(); SHARDS].map(|()| CacheAligned(Lock::new(value()))),
            ));
        }

        Sharded::Single(Lock::new(value()))
    }

    /// The shard is selected by hashing `val` with `FxHasher`.
    #[inline]
    pub fn get_shard_by_value<K: Hash + ?Sized>(&self, val: &K) -> &Lock<T> {
        match self {
            Self::Single(single) => single,
            Self::Shards(..) => self.get_shard_by_hash(make_hash(val)),
        }
    }

    #[inline]
    pub fn get_shard_by_hash(&self, hash: u64) -> &Lock<T> {
        self.get_shard_by_index(get_shard_hash(hash))
    }

    #[inline]
    pub fn get_shard_by_index(&self, i: usize) -> &Lock<T> {
        match self {
            Self::Single(single) => single,
            Self::Shards(shards) => {
                // SAFETY: The index gets ANDed with the shard mask, ensuring it is always inbounds.
                unsafe { &shards.get_unchecked(i & (SHARDS - 1)).0 }
            }
        }
    }

    /// The shard is selected by hashing `val` with `FxHasher`.
    #[inline]
    #[track_caller]
    pub fn lock_shard_by_value<K: Hash + ?Sized>(&self, val: &K) -> LockGuard<'_, T> {
        match self {
            Self::Single(single) => {
                // Synchronization is disabled so use the `lock_assume_no_sync` method optimized
                // for that case.

                // SAFETY: We know `is_dyn_thread_safe` was false when creating the lock thus
                // `might_be_dyn_thread_safe` was also false.
                unsafe { single.lock_assume(Mode::NoSync) }
            }
            Self::Shards(..) => self.lock_shard_by_hash(make_hash(val)),
        }
    }

    #[inline]
    #[track_caller]
    pub fn lock_shard_by_hash(&self, hash: u64) -> LockGuard<'_, T> {
        self.lock_shard_by_index(get_shard_hash(hash))
    }

    #[inline]
    #[track_caller]
    pub fn lock_shard_by_index(&self, i: usize) -> LockGuard<'_, T> {
        match self {
            Self::Single(single) => {
                // Synchronization is disabled so use the `lock_assume_no_sync` method optimized
                // for that case.

                // SAFETY: We know `is_dyn_thread_safe` was false when creating the lock thus
                // `might_be_dyn_thread_safe` was also false.
                unsafe { single.lock_assume(Mode::NoSync) }
            }
            Self::Shards(shards) => {
                // Synchronization is enabled so use the `lock_assume_sync` method optimized
                // for that case.

                // SAFETY (get_unchecked): The index gets ANDed with the shard mask, ensuring it is
                // always inbounds.
                // SAFETY (lock_assume_sync): We know `is_dyn_thread_safe` was true when creating
                // the lock thus `might_be_dyn_thread_safe` was also true.
                unsafe { shards.get_unchecked(i & (SHARDS - 1)).0.lock_assume(Mode::Sync) }
            }
        }
    }

    #[inline]
    pub fn lock_shards(&self) -> impl Iterator<Item = LockGuard<'_, T>> {
        match self {
            Self::Single(single) => Either::Left(iter::once(single.lock())),
            Self::Shards(shards) => Either::Right(shards.iter().map(|shard| shard.0.lock())),
        }
    }

    #[inline]
    pub fn try_lock_shards(&self) -> impl Iterator<Item = Option<LockGuard<'_, T>>> {
        match self {
            Self::Single(single) => Either::Left(iter::once(single.try_lock())),
            Self::Shards(shards) => Either::Right(shards.iter().map(|shard| shard.0.try_lock())),
        }
    }
}

#[inline]
pub fn shards() -> usize {
    if is_dyn_thread_safe() {
        return SHARDS;
    }

    1
}

pub type ShardedHashMap<K, V> = Sharded<HashTable<(K, V)>>;

impl<K: Eq, V> ShardedHashMap<K, V> {
    pub fn with_capacity(cap: usize) -> Self {
        Self::new(|| HashTable::with_capacity(cap))
    }
    pub fn len(&self) -> usize {
        self.lock_shards().map(|shard| shard.len()).sum()
    }
}

impl<K: Eq + Hash, V> ShardedHashMap<K, V> {
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
        V: Clone,
    {
        let hash = make_hash(key);
        let shard = self.lock_shard_by_hash(hash);
        let (_, value) = shard.find(hash, |(k, _)| k.borrow() == key)?;
        Some(value.clone())
    }

    #[inline]
    pub fn get_or_insert_with(&self, key: K, default: impl FnOnce() -> V) -> V
    where
        V: Copy,
    {
        let hash = make_hash(&key);
        let mut shard = self.lock_shard_by_hash(hash);

        match table_entry(&mut shard, hash, &key) {
            Entry::Occupied(e) => e.get().1,
            Entry::Vacant(e) => {
                let value = default();
                e.insert((key, value));
                value
            }
        }
    }

    #[inline]
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let hash = make_hash(&key);
        let mut shard = self.lock_shard_by_hash(hash);

        match table_entry(&mut shard, hash, &key) {
            Entry::Occupied(e) => {
                let previous = mem::replace(&mut e.into_mut().1, value);
                Some(previous)
            }
            Entry::Vacant(e) => {
                e.insert((key, value));
                None
            }
        }
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
        let mut shard = self.lock_shard_by_hash(hash);

        match table_entry(&mut shard, hash, value) {
            Entry::Occupied(e) => e.get().0,
            Entry::Vacant(e) => {
                let v = make();
                e.insert((v, ()));
                v
            }
        }
    }

    #[inline]
    pub fn intern<Q>(&self, value: Q, make: impl FnOnce(Q) -> K) -> K
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash(&value);
        let mut shard = self.lock_shard_by_hash(hash);

        match table_entry(&mut shard, hash, &value) {
            Entry::Occupied(e) => e.get().0,
            Entry::Vacant(e) => {
                let v = make(value);
                e.insert((v, ()));
                v
            }
        }
    }
}

pub trait IntoPointer {
    /// Returns a pointer which outlives `self`.
    fn into_pointer(&self) -> *const ();
}

impl<K: Eq + Hash + Copy + IntoPointer> ShardedHashMap<K, ()> {
    pub fn contains_pointer_to<T: Hash + IntoPointer>(&self, value: &T) -> bool {
        let hash = make_hash(&value);
        let shard = self.lock_shard_by_hash(hash);
        let value = value.into_pointer();
        shard.find(hash, |(k, ())| k.into_pointer() == value).is_some()
    }
}

#[inline]
pub fn make_hash<K: Hash + ?Sized>(val: &K) -> u64 {
    let mut state = FxHasher::default();
    val.hash(&mut state);
    state.finish()
}

#[inline]
fn table_entry<'a, K, V, Q>(
    table: &'a mut HashTable<(K, V)>,
    hash: u64,
    key: &Q,
) -> Entry<'a, (K, V)>
where
    K: Hash + Borrow<Q>,
    Q: ?Sized + Eq,
{
    table.entry(hash, move |(k, _)| k.borrow() == key, |(k, _)| make_hash(k))
}

/// Get a shard with a pre-computed hash value. If `get_shard_by_value` is
/// ever used in combination with `get_shard_by_hash` on a single `Sharded`
/// instance, then `hash` must be computed with `FxHasher`. Otherwise,
/// `hash` can be computed with any hasher, so long as that hasher is used
/// consistently for each `Sharded` instance.
#[inline]
fn get_shard_hash(hash: u64) -> usize {
    let hash_len = size_of::<usize>();
    // Ignore the top 7 bits as hashbrown uses these and get the next SHARD_BITS highest bits.
    // hashbrown also uses the lowest bits, so we can't use those
    (hash >> (hash_len * 8 - 7 - SHARD_BITS)) as usize
}
