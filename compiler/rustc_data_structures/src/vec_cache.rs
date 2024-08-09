//! VecCache maintains a mapping from K -> (V, I) pairing. K and I must be roughly u32-sized, and V
//! must be Copy.
//!
//! VecCache supports efficient concurrent put/get across the key space, with write-once semantics
//! (i.e., a given key can only be put once). Subsequent puts will panic.
//!
//! This is currently used for query caching.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::offset_of;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use rustc_index::Idx;

use crate::sync::AtomicUsize;

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

#[derive(Copy, Clone, Debug)]
struct SlotIndex {
    bucket_idx: usize,
    entries: usize,
    index_in_bucket: usize,
}

impl SlotIndex {
    #[inline]
    fn from_index(idx: u32) -> Self {
        let mut bucket = idx.checked_ilog2().unwrap_or(0) as usize;
        let entries;
        let running_sum;
        if bucket <= 11 {
            entries = 1 << 12;
            running_sum = 0;
            bucket = 0;
        } else {
            entries = 1 << bucket;
            running_sum = entries;
            bucket = bucket - 11;
        }
        SlotIndex { bucket_idx: bucket, entries, index_in_bucket: idx as usize - running_sum }
    }

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
        // SAFETY: Follows from preconditions on `buckets` and `self`.
        let slot = unsafe { ptr.add(self.index_in_bucket) };

        // SAFETY:
        let index_and_lock =
            unsafe { &*slot.byte_add(offset_of!(Slot<V>, index_and_lock)).cast::<AtomicU32>() };
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
        let value = unsafe { slot.byte_add(offset_of!(Slot<V>, value)).cast::<V>().read() };
        Some((value, index))
    }

    /// Returns true if this successfully put into the map.
    #[inline]
    fn put<V>(&self, buckets: &[AtomicPtr<Slot<V>>; 21], value: V, extra: u32) -> bool {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

        // SAFETY: `bucket_idx` is ilog2(u32).saturating_sub(11), which is at most 21, i.e.,
        // in-bounds of buckets.
        let bucket = unsafe { buckets.get_unchecked(self.bucket_idx) };
        let mut ptr = bucket.load(Ordering::Acquire);
        let _allocator_guard;
        if ptr.is_null() {
            // If we load null, then acquire the global lock; this path is quite cold, so it's cheap
            // to use a global lock.
            _allocator_guard = LOCK.lock();
            // And re-load the value.
            ptr = bucket.load(Ordering::Acquire);
        }

        // OK, now under the allocator lock, if we're still null then it's definitely us that will
        // initialize this bucket.
        if ptr.is_null() {
            let bucket_layout =
                std::alloc::Layout::array::<Slot<V>>(self.entries as usize).unwrap();
            // SAFETY: Always >0 entries in each bucket.
            let allocated = unsafe { std::alloc::alloc_zeroed(bucket_layout).cast::<Slot<V>>() };
            if allocated.is_null() {
                std::alloc::handle_alloc_error(bucket_layout);
            }
            bucket.store(allocated, Ordering::Release);
            ptr = allocated;
        }
        assert!(!ptr.is_null());

        // SAFETY: `index_in_bucket` is always in-bounds of the allocated array.
        assert!(self.index_in_bucket < self.entries);
        let slot = unsafe { ptr.add(self.index_in_bucket) };

        let index_and_lock =
            unsafe { &*slot.byte_add(offset_of!(Slot<V>, index_and_lock)).cast::<AtomicU32>() };
        match index_and_lock.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                // We have acquired the initialization lock. It is our job to write `value` and
                // then set the lock to the real index.

                unsafe {
                    slot.byte_add(offset_of!(Slot<V>, value)).cast::<V>().write(value);
                }

                index_and_lock.store(extra.checked_add(2).unwrap(), Ordering::Release);

                true
            }

            // Treat "initializing" as actually initialized: we lost the race and should skip
            // any updates to this slot. In practice this should be unreachable since we're guarded
            // by an external lock that only allows one initialization for any given query result.
            Err(1) => unreachable!(),

            // This slot was already populated. Also ignore, currently this is the same as
            // "initializing".
            Err(_) => false,
        }
    }
}

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

    // Present and len are only used during incremental and self-profiling.
    // They are an optimization over iterating the full buckets array.
    present: [AtomicPtr<Slot<()>>; 21],
    len: AtomicUsize,

    key: PhantomData<(K, I)>,
}

// SAFETY: No access to `V` is made.
unsafe impl<K: Idx, #[may_dangle] V, I> Drop for VecCache<K, V, I> {
    fn drop(&mut self) {
        // We have unique ownership, so no locks etc. are needed. Since `K` and `V` are both `Copy`,
        // we are also guaranteed to just need to deallocate any large arrays (not iterate over
        // contents).

        let mut entries = 1 << 12;
        for bucket in self.buckets.iter() {
            let bucket = bucket.load(Ordering::Acquire);
            if !bucket.is_null() {
                let layout = std::alloc::Layout::array::<Slot<V>>(entries).unwrap();
                unsafe {
                    std::alloc::dealloc(bucket.cast(), layout);
                }
            }
            entries *= 2;
        }

        let mut entries = 1 << 12;
        for bucket in self.present.iter() {
            let bucket = bucket.load(Ordering::Acquire);
            if !bucket.is_null() {
                let layout = std::alloc::Layout::array::<Slot<V>>(entries).unwrap();
                unsafe {
                    std::alloc::dealloc(bucket.cast(), layout);
                }
            }
            entries *= 2;
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
