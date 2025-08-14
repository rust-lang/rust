//! VecCache maintains a mapping from K -> (V, I) pairing. K and I must be roughly u32-sized, and V
//! must be Copy.
//!
//! VecCache supports efficient concurrent put/get across the key space, with write-once semantics
//! (i.e., a given key can only be put once). Subsequent puts will panic.
//!
//! This is currently used for query caching.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};

use rustc_index::Idx;

struct Slot<V> {
    // We never construct &Slot<V> so it's fine for this to not be in an UnsafeCell.
    value: V,
    // This is both an index and a once-lock.
    //
    // 0: not yet initialized.
    // 1: lock held, initializing.
    // 2..u32::MAX - 2: initialized.
    index_and_lock: AtomicU32,
}

/// This uniquely identifies a single `Slot<V>` entry in the buckets map, and provides accessors for
/// either getting the value or putting a value.
#[derive(Copy, Clone, Debug)]
struct SlotIndex {
    // the index of the bucket in VecCache (0 to 20)
    bucket_idx: usize,
    // number of entries in that bucket
    entries: usize,
    // the index of the slot within the bucket
    index_in_bucket: usize,
}

// This makes sure the counts are consistent with what we allocate, precomputing each bucket a
// compile-time. Visiting all powers of two is enough to hit all the buckets.
//
// We confirm counts are accurate in the slot_index_exhaustive test.
const ENTRIES_BY_BUCKET: [usize; 21] = {
    let mut entries = [0; 21];
    let mut key = 0;
    loop {
        let si = SlotIndex::from_index(key);
        entries[si.bucket_idx] = si.entries;
        if key == 0 {
            key = 1;
        } else if key == (1 << 31) {
            break;
        } else {
            key <<= 1;
        }
    }
    entries
};

impl SlotIndex {
    // This unpacks a flat u32 index into identifying which bucket it belongs to and the offset
    // within that bucket. As noted in the VecCache docs, buckets double in size with each index.
    // Typically that would mean 31 buckets (2^0 + 2^1 ... + 2^31 = u32::MAX - 1), but to reduce
    // the size of the VecCache struct and avoid uselessly small allocations, we instead have the
    // first bucket have 2**12 entries. To simplify the math, the second bucket also 2**12 entries,
    // and buckets double from there.
    //
    // We assert that [0, 2**32 - 1] uniquely map through this function to individual, consecutive
    // slots (see `slot_index_exhaustive` in tests).
    #[inline]
    const fn from_index(idx: u32) -> Self {
        const FIRST_BUCKET_SHIFT: usize = 12;
        if idx < (1 << FIRST_BUCKET_SHIFT) {
            return SlotIndex {
                bucket_idx: 0,
                entries: 1 << FIRST_BUCKET_SHIFT,
                index_in_bucket: idx as usize,
            };
        }
        // We already ruled out idx 0, so this `ilog2` never panics (and the check optimizes away)
        let bucket = idx.ilog2() as usize;
        let entries = 1 << bucket;
        SlotIndex {
            bucket_idx: bucket - FIRST_BUCKET_SHIFT + 1,
            entries,
            index_in_bucket: idx as usize - entries,
        }
    }

    // SAFETY: Buckets must be managed solely by functions here (i.e., get/put on SlotIndex) and
    // `self` comes from SlotIndex::from_index
    #[inline]
    unsafe fn get<V: Copy>(&self, buckets: &[AtomicPtr<Slot<V>>; 21]) -> Option<(V, u32)> {
        // SAFETY: `bucket_idx` is ilog2(u32).saturating_sub(11), which is at most 21, i.e.,
        // in-bounds of buckets. See `from_index` for computation.
        let bucket = unsafe { buckets.get_unchecked(self.bucket_idx) };
        let ptr = bucket.load(Ordering::Acquire);
        // Bucket is not yet initialized: then we obviously won't find this entry in that bucket.
        if ptr.is_null() {
            return None;
        }
        assert!(self.index_in_bucket < self.entries);
        // SAFETY: `bucket` was allocated (so <= isize in total bytes) to hold `entries`, so this
        // must be inbounds.
        let slot = unsafe { ptr.add(self.index_in_bucket) };

        // SAFETY: initialized bucket has zeroed all memory within the bucket, so we are valid for
        // AtomicU32 access.
        let index_and_lock = unsafe { &(*slot).index_and_lock };
        let current = index_and_lock.load(Ordering::Acquire);
        let index = match current {
            0 => return None,
            // Treat "initializing" as actually just not initialized at all.
            // The only reason this is a separate state is that `complete` calls could race and
            // we can't allow that, but from load perspective there's no difference.
            1 => return None,
            _ => current - 2,
        };

        // SAFETY:
        // * slot is a valid pointer (buckets are always valid for the index we get).
        // * value is initialized since we saw a >= 2 index above.
        // * `V: Copy`, so safe to read.
        let value = unsafe { (*slot).value };
        Some((value, index))
    }

    fn bucket_ptr<V>(&self, bucket: &AtomicPtr<Slot<V>>) -> *mut Slot<V> {
        let ptr = bucket.load(Ordering::Acquire);
        if ptr.is_null() { self.initialize_bucket(bucket) } else { ptr }
    }

    #[cold]
    fn initialize_bucket<V>(&self, bucket: &AtomicPtr<Slot<V>>) -> *mut Slot<V> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

        // If we are initializing the bucket, then acquire a global lock.
        //
        // This path is quite cold, so it's cheap to use a global lock. This ensures that we never
        // have multiple allocations for the same bucket.
        let _allocator_guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let ptr = bucket.load(Ordering::Acquire);

        // OK, now under the allocator lock, if we're still null then it's definitely us that will
        // initialize this bucket.
        if ptr.is_null() {
            let bucket_layout =
                std::alloc::Layout::array::<Slot<V>>(self.entries as usize).unwrap();
            // This is more of a sanity check -- this code is very cold, so it's safe to pay a
            // little extra cost here.
            assert!(bucket_layout.size() > 0);
            // SAFETY: Just checked that size is non-zero.
            let allocated = unsafe { std::alloc::alloc_zeroed(bucket_layout).cast::<Slot<V>>() };
            if allocated.is_null() {
                std::alloc::handle_alloc_error(bucket_layout);
            }
            bucket.store(allocated, Ordering::Release);
            allocated
        } else {
            // Otherwise some other thread initialized this bucket after we took the lock. In that
            // case, just return early.
            ptr
        }
    }

    /// Returns true if this successfully put into the map.
    #[inline]
    fn put<V>(&self, buckets: &[AtomicPtr<Slot<V>>; 21], value: V, extra: u32) -> bool {
        // SAFETY: `bucket_idx` is ilog2(u32).saturating_sub(11), which is at most 21, i.e.,
        // in-bounds of buckets.
        let bucket = unsafe { buckets.get_unchecked(self.bucket_idx) };
        let ptr = self.bucket_ptr(bucket);

        assert!(self.index_in_bucket < self.entries);
        // SAFETY: `bucket` was allocated (so <= isize in total bytes) to hold `entries`, so this
        // must be inbounds.
        let slot = unsafe { ptr.add(self.index_in_bucket) };

        // SAFETY: initialized bucket has zeroed all memory within the bucket, so we are valid for
        // AtomicU32 access.
        let index_and_lock = unsafe { &(*slot).index_and_lock };
        match index_and_lock.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                // We have acquired the initialization lock. It is our job to write `value` and
                // then set the lock to the real index.

                unsafe {
                    (&raw mut (*slot).value).write(value);
                }

                index_and_lock.store(extra.checked_add(2).unwrap(), Ordering::Release);

                true
            }

            // Treat "initializing" as the caller's fault. Callers are responsible for ensuring that
            // there are no races on initialization. In the compiler's current usage for query
            // caches, that's the "active query map" which ensures each query actually runs once
            // (even if concurrently started).
            Err(1) => panic!("caller raced calls to put()"),

            // This slot was already populated. Also ignore, currently this is the same as
            // "initializing".
            Err(_) => false,
        }
    }
}

/// In-memory cache for queries whose keys are densely-numbered IDs
/// (e.g `CrateNum`, `LocalDefId`), and can therefore be used as indices
/// into a dense vector of cached values.
///
/// (As of [#124780] the underlying storage is not an actual `Vec`, but rather
/// a series of increasingly-large buckets, for improved performance when the
/// parallel frontend is using multiple threads.)
///
/// Each entry in the cache stores the query's return value (`V`), and also
/// an associated index (`I`), which in practice is a `DepNodeIndex` used for
/// query dependency tracking.
///
/// [#124780]: https://github.com/rust-lang/rust/pull/124780
pub struct VecCache<K: Idx, V, I> {
    // Entries per bucket:
    // Bucket  0:       4096 2^12
    // Bucket  1:       4096 2^12
    // Bucket  2:       8192
    // Bucket  3:      16384
    // ...
    // Bucket 19: 1073741824
    // Bucket 20: 2147483648
    // The total number of entries if all buckets are initialized is u32::MAX-1.
    buckets: [AtomicPtr<Slot<V>>; 21],

    // In the compiler's current usage these are only *read* during incremental and self-profiling.
    // They are an optimization over iterating the full buckets array.
    present: [AtomicPtr<Slot<()>>; 21],
    len: AtomicUsize,

    key: PhantomData<(K, I)>,
}

impl<K: Idx, V, I> Default for VecCache<K, V, I> {
    fn default() -> Self {
        VecCache {
            buckets: Default::default(),
            key: PhantomData,
            len: Default::default(),
            present: Default::default(),
        }
    }
}

// SAFETY: No access to `V` is made.
unsafe impl<K: Idx, #[may_dangle] V, I> Drop for VecCache<K, V, I> {
    fn drop(&mut self) {
        // We have unique ownership, so no locks etc. are needed. Since `K` and `V` are both `Copy`,
        // we are also guaranteed to just need to deallocate any large arrays (not iterate over
        // contents).
        //
        // Confirm no need to deallocate individual entries. Note that `V: Copy` is asserted on
        // insert/lookup but not necessarily construction, primarily to avoid annoyingly propagating
        // the bounds into struct definitions everywhere.
        assert!(!std::mem::needs_drop::<K>());
        assert!(!std::mem::needs_drop::<V>());

        for (idx, bucket) in self.buckets.iter().enumerate() {
            let bucket = bucket.load(Ordering::Acquire);
            if !bucket.is_null() {
                let layout = std::alloc::Layout::array::<Slot<V>>(ENTRIES_BY_BUCKET[idx]).unwrap();
                unsafe {
                    std::alloc::dealloc(bucket.cast(), layout);
                }
            }
        }

        for (idx, bucket) in self.present.iter().enumerate() {
            let bucket = bucket.load(Ordering::Acquire);
            if !bucket.is_null() {
                let layout = std::alloc::Layout::array::<Slot<()>>(ENTRIES_BY_BUCKET[idx]).unwrap();
                unsafe {
                    std::alloc::dealloc(bucket.cast(), layout);
                }
            }
        }
    }
}

impl<K, V, I> VecCache<K, V, I>
where
    K: Eq + Idx + Copy + Debug,
    V: Copy,
    I: Idx + Copy,
{
    #[inline(always)]
    pub fn lookup(&self, key: &K) -> Option<(V, I)> {
        let key = u32::try_from(key.index()).unwrap();
        let slot_idx = SlotIndex::from_index(key);
        match unsafe { slot_idx.get(&self.buckets) } {
            Some((value, idx)) => Some((value, I::new(idx as usize))),
            None => None,
        }
    }

    #[inline]
    pub fn complete(&self, key: K, value: V, index: I) {
        let key = u32::try_from(key.index()).unwrap();
        let slot_idx = SlotIndex::from_index(key);
        if slot_idx.put(&self.buckets, value, index.index() as u32) {
            let present_idx = self.len.fetch_add(1, Ordering::Relaxed);
            let slot = SlotIndex::from_index(present_idx as u32);
            // We should always be uniquely putting due to `len` fetch_add returning unique values.
            assert!(slot.put(&self.present, (), key));
        }
    }

    pub fn iter(&self, f: &mut dyn FnMut(&K, &V, I)) {
        for idx in 0..self.len.load(Ordering::Acquire) {
            let key = SlotIndex::from_index(idx as u32);
            match unsafe { key.get(&self.present) } {
                // This shouldn't happen in our current usage (iter is really only
                // used long after queries are done running), but if we hit this in practice it's
                // probably fine to just break early.
                None => unreachable!(),
                Some(((), key)) => {
                    let key = K::new(key as usize);
                    // unwrap() is OK: present entries are always written only after we put the real
                    // entry.
                    let value = self.lookup(&key).unwrap();
                    f(&key, &value.0, value.1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
