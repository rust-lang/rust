use std::hash::{Hasher, Hash};
use std::mem;
use std::borrow::Borrow;
use std::collections::hash_map::RawEntryMut;
use smallvec::SmallVec;
use crate::fx::{FxHasher, FxHashMap};
use crate::sync::{Lock, LockGuard};

#[derive(Clone, Default)]
#[cfg_attr(parallel_compiler, repr(align(64)))]
struct CacheAligned<T>(T);

#[cfg(parallel_compiler)]
// 32 shards is sufficient to reduce contention on an 8-core Ryzen 7 1700,
// but this should be tested on higher core count CPUs. How the `Sharded` type gets used
// may also affect the ideal nunber of shards.
const SHARD_BITS: usize = 5;

#[cfg(not(parallel_compiler))]
const SHARD_BITS: usize = 0;

pub const SHARDS: usize = 1 << SHARD_BITS;

/// An array of cache-line aligned inner locked structures with convenience methods.
#[derive(Clone)]
pub struct Sharded<T> {
    shards: [CacheAligned<Lock<T>>; SHARDS],
}

impl<T: Default> Default for Sharded<T> {
    #[inline]
    fn default() -> Self {
        Self::new(|| T::default())
    }
}

impl<T> Sharded<T> {
    #[inline]
    pub fn new(mut value: impl FnMut() -> T) -> Self {
        // Create a vector of the values we want
        let mut values: SmallVec<[_; SHARDS]> = (0..SHARDS).map(|_| {
            CacheAligned(Lock::new(value()))
        }).collect();

        // Create an unintialized array
        let mut shards: mem::MaybeUninit<[CacheAligned<Lock<T>>; SHARDS]> =
            mem::MaybeUninit::uninit();

        unsafe {
            // Copy the values into our array
            let first = shards.as_mut_ptr() as *mut CacheAligned<Lock<T>>;
            values.as_ptr().copy_to_nonoverlapping(first, SHARDS);

            // Ignore the content of the vector
            values.set_len(0);

            Sharded {
                shards: shards.assume_init(),
            }
        }
    }

    #[inline]
    pub fn get_shard_by_value<K: Hash + ?Sized>(&self, val: &K) -> &Lock<T> {
        if SHARDS == 1 {
            &self.shards[0].0
        } else {
            self.get_shard_by_hash(make_hash(val))
        }
    }

    #[inline]
    pub fn get_shard_by_hash(&self, hash: u64) -> &Lock<T> {
        let hash_len = mem::size_of::<usize>();
        // Ignore the top 7 bits as hashbrown uses these and get the next SHARD_BITS highest bits.
        // hashbrown also uses the lowest bits, so we can't use those
        let bits = (hash >> (hash_len * 8 - 7 - SHARD_BITS)) as usize;
        let i = bits % SHARDS;
        &self.shards[i].0
    }

    pub fn lock_shards(&self) -> Vec<LockGuard<'_, T>> {
        (0..SHARDS).map(|i| self.shards[i].0.lock()).collect()
    }

    pub fn try_lock_shards(&self) -> Option<Vec<LockGuard<'_, T>>> {
        (0..SHARDS).map(|i| self.shards[i].0.try_lock()).collect()
    }
}

pub type ShardedHashMap<K, V> = Sharded<FxHashMap<K, V>>;

impl<K: Eq + Hash, V> ShardedHashMap<K, V> {
    pub fn len(&self) -> usize {
        self.lock_shards().iter().map(|shard| shard.len()).sum()
    }
}

impl<K: Eq + Hash + Copy> ShardedHashMap<K, ()> {
    #[inline]
    pub fn intern_ref<Q: ?Sized>(&self, value: &Q, make: impl FnOnce() -> K) -> K
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        let hash = make_hash(value);
        let mut shard = self.get_shard_by_hash(hash).lock();
        let entry = shard.raw_entry_mut().from_key_hashed_nocheck(hash, value);

        match entry {
            RawEntryMut::Occupied(e) => *e.key(),
            RawEntryMut::Vacant(e) => {
                let v = make();
                e.insert_hashed_nocheck(hash, v, ());
                v
            }
        }
    }

    #[inline]
    pub fn intern<Q>(&self, value: Q, make: impl FnOnce(Q) -> K) -> K
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        let hash = make_hash(&value);
        let mut shard = self.get_shard_by_hash(hash).lock();
        let entry = shard.raw_entry_mut().from_key_hashed_nocheck(hash, &value);

        match entry {
            RawEntryMut::Occupied(e) => *e.key(),
            RawEntryMut::Vacant(e) => {
                let v = make(value);
                e.insert_hashed_nocheck(hash, v, ());
                v
            }
        }
    }
}

#[inline]
fn make_hash<K: Hash + ?Sized>(val: &K) -> u64 {
    let mut state = FxHasher::default();
    val.hash(&mut state);
    state.finish()
}
