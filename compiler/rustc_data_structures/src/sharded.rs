use crate::fx::{FxHashMap, FxHasher};
use crate::sync::LockLike;
use parking_lot::{Mutex, MutexGuard};
use std::borrow::Borrow;
use std::cell::{RefCell, RefMut};
use std::collections::hash_map::RawEntryMut;
use std::hash::{Hash, Hasher};
use std::mem;

pub trait Shard {
    type Impl<T>: ShardImpl<T>;
}

pub trait ShardImpl<T> {
    type Lock: LockLike<T>;

    fn new(value: impl FnMut() -> T) -> Self;

    fn get_shard_by_value<K: Hash + ?Sized>(&self, _val: &K) -> &Self::Lock;

    fn get_shard_by_hash(&self, _hash: u64) -> &Self::Lock;

    fn lock_shards(&self) -> Vec<<Self::Lock as LockLike<T>>::LockGuard<'_>>;

    fn try_lock_shards(&self) -> Option<Vec<<Self::Lock as LockLike<T>>::LockGuard<'_>>>;
}

#[derive(Default)]
pub struct SingleShard;

impl Shard for SingleShard {
    type Impl<T> = SingleShardImpl<T>;
}

/// An array of cache-line aligned inner locked structures with convenience methods.
pub struct SingleShardImpl<T> {
    shard: RefCell<T>,
}

impl<T: Default> Default for SingleShardImpl<T> {
    #[inline]
    fn default() -> Self {
        Self { shard: RefCell::new(T::default()) }
    }
}

impl<T> ShardImpl<T> for SingleShardImpl<T> {
    type Lock = RefCell<T>;

    #[inline]
    fn new(mut value: impl FnMut() -> T) -> Self {
        SingleShardImpl { shard: RefCell::new(value()) }
    }

    #[inline]
    fn get_shard_by_value<K: Hash + ?Sized>(&self, _val: &K) -> &RefCell<T> {
        &self.shard
    }

    #[inline]
    fn get_shard_by_hash(&self, _hash: u64) -> &RefCell<T> {
        &self.shard
    }

    fn lock_shards(&self) -> Vec<RefMut<'_, T>> {
        vec![self.shard.lock()]
    }

    fn try_lock_shards(&self) -> Option<Vec<RefMut<'_, T>>> {
        Some(vec![self.shard.try_lock()?])
    }
}

const SHARD_BITS: usize = 5;

pub const SHARDS: usize = 1 << SHARD_BITS;

#[derive(Default)]
pub struct Sharded;

impl Shard for Sharded {
    type Impl<T> = ShardedImpl<T>;
}

#[derive(Default)]
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

pub struct ShardedImpl<T> {
    shards: [CacheAligned<Mutex<T>>; SHARDS],
}

impl<T: Default> Default for ShardedImpl<T> {
    #[inline]
    fn default() -> Self {
        Self::new(T::default)
    }
}

impl<T> ShardImpl<T> for ShardedImpl<T> {
    type Lock = Mutex<T>;

    #[inline]
    fn new(mut value: impl FnMut() -> T) -> Self {
        ShardedImpl { shards: [(); SHARDS].map(|()| CacheAligned(Mutex::new(value()))) }
    }

    /// The shard is selected by hashing `val` with `FxHasher`.
    #[inline]
    fn get_shard_by_value<K: Hash + ?Sized>(&self, val: &K) -> &Mutex<T> {
        self.get_shard_by_hash(make_hash(val))
    }

    #[inline]
    fn get_shard_by_hash(&self, hash: u64) -> &Mutex<T> {
        &self.shards[get_shard_index_by_hash(hash)].0
    }

    fn lock_shards(&self) -> Vec<MutexGuard<'_, T>> {
        (0..SHARDS).map(|i| self.shards[i].0.lock()).collect()
    }

    fn try_lock_shards(&self) -> Option<Vec<MutexGuard<'_, T>>> {
        (0..SHARDS).map(|i| self.shards[i].0.try_lock()).collect()
    }
}

pub struct DynSharded<T> {
    single_thread: bool,
    single_shard: RefCell<T>,
    parallel_shard: ShardedImpl<T>,
}

// just for speed test
unsafe impl<T> Sync for DynSharded<T> {}

impl<T: Default> Default for DynSharded<T> {
    #[inline]
    fn default() -> Self {
        let single_thread = !crate::sync::active();
        DynSharded {
            single_thread,
            single_shard: RefCell::new(T::default()),
            parallel_shard: ShardedImpl::default(),
        }
    }
}

impl<T: Default> DynSharded<T> {
    pub fn new(mut value: impl FnMut() -> T) -> Self {
        if !crate::sync::active() {
            DynSharded {
                single_thread: true,
                single_shard: RefCell::new(value()),
                parallel_shard: ShardedImpl::default(),
            }
        } else {
            DynSharded {
                single_thread: false,
                single_shard: RefCell::new(T::default()),
                parallel_shard: ShardedImpl::new(value),
            }
        }
    }

    /// The shard is selected by hashing `val` with `FxHasher`.
    #[inline]
    pub fn with_get_shard_by_value<K: Hash + ?Sized, F: FnOnce(&mut T) -> R, R>(
        &self,
        val: &K,
        f: F,
    ) -> R {
        if self.single_thread {
            let mut lock = self.single_shard.borrow_mut();
            f(&mut *lock)
        } else {
            let mut lock = self.parallel_shard.get_shard_by_value(val).lock();
            f(&mut *lock)
        }
    }

    #[inline]
    pub fn with_get_shard_by_hash<F: FnOnce(&mut T) -> R, R>(&self, hash: u64, f: F) -> R {
        if self.single_thread {
            let mut lock = self.single_shard.borrow_mut();
            f(&mut *lock)
        } else {
            let mut lock = self.parallel_shard.get_shard_by_hash(hash).lock();
            f(&mut *lock)
        }
    }

    #[inline]
    pub fn with_lock_shards<F: FnMut(&mut T) -> R, R>(&self, mut f: F) -> Vec<R> {
        if self.single_thread {
            let mut lock = self.single_shard.borrow_mut();
            vec![f(&mut *lock)]
        } else {
            (0..SHARDS).map(|i| f(&mut *self.parallel_shard.shards[i].0.lock())).collect()
        }
    }

    #[inline]
    pub fn with_try_lock_shards<F: FnMut(&mut T) -> R, R>(&self, mut f: F) -> Option<Vec<R>> {
        if self.single_thread {
            let mut lock = self.single_shard.try_borrow_mut().ok()?;
            Some(vec![f(&mut *lock)])
        } else {
            (0..SHARDS)
                .map(|i| {
                    let mut shard = self.parallel_shard.shards[i].0.try_lock()?;
                    Some(f(&mut *shard))
                })
                .collect()
        }
    }

    #[inline]
    pub fn get_lock_by_value<K: Hash + ?Sized>(&self, val: &K) -> &Mutex<T> {
        self.parallel_shard.get_shard_by_value(val)
    }

    #[inline]
    pub fn get_borrow_by_value<K: Hash + ?Sized>(&self, _val: &K) -> &RefCell<T> {
        &self.single_shard
    }
}

pub type ShardedHashMap<K, V> = DynSharded<FxHashMap<K, V>>;

impl<K: Eq, V> ShardedHashMap<K, V> {
    pub fn len(&self) -> usize {
        self.with_lock_shards(|shard| shard.len()).into_iter().sum()
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

/// Get a shard with a pre-computed hash value. If `get_shard_by_value` is
/// ever used in combination with `get_shard_by_hash` on a single `Sharded`
/// instance, then `hash` must be computed with `FxHasher`. Otherwise,
/// `hash` can be computed with any hasher, so long as that hasher is used
/// consistently for each `Sharded` instance.
#[inline]
#[allow(clippy::modulo_one)]
pub fn get_shard_index_by_hash(hash: u64) -> usize {
    let hash_len = mem::size_of::<usize>();
    // Ignore the top 7 bits as hashbrown uses these and get the next SHARD_BITS highest bits.
    // hashbrown also uses the lowest bits, so we can't use those
    let bits = (hash >> (hash_len * 8 - 7 - SHARD_BITS)) as usize;
    bits % SHARDS
}
