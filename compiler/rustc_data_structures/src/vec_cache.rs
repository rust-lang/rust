//! VecCache maintains a mapping from K -> (V, I) pairing. K and I must be roughly u32-sized, and V
//! must be Copy.
//!
//! VecCache supports efficient concurrent put/get across the key space, with write-once semantics
//! (i.e., a given key can only be put once). Subsequent puts will panic.
//!
//! This is currently used for query caching.

use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr::{drop_in_place, slice_from_raw_parts_mut};
use std::slice;
use std::sync::atomic::{AtomicPtr, Ordering};

use rustc_index::Idx;

use crate::cache_entry::CacheEntry;

#[cfg(test)]
mod tests;

/// This uniquely identifies a single `Slot<V>` entry in the buckets map, and provides accessors for
/// either getting the value or putting a value.
#[derive(Copy, Clone, Debug)]
struct SlotIndex {
    // the index of the bucket in VecCache (0 to 20)
    bucket_idx: BucketIndex,
    // the index of the slot within the bucket
    index_in_bucket: usize,
}

// This makes sure the counts are consistent with what we allocate, precomputing each bucket a
// compile-time. Visiting all powers of two is enough to hit all the buckets.
//
// We confirm counts are accurate in the slot_index_exhaustive test.
const ENTRIES_BY_BUCKET: [usize; BUCKETS] = {
    let mut entries = [0; BUCKETS];
    let mut key = 0;
    loop {
        let si = SlotIndex::from_index(key);
        entries[si.bucket_idx.to_usize()] = si.bucket_idx.capacity();
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

const BUCKETS: usize = 21;

impl SlotIndex {
    /// Unpacks a flat 32-bit index into a [`BucketIndex`] and a slot offset within that bucket.
    #[inline]
    const fn from_index(idx: u32) -> Self {
        let (bucket_idx, index_in_bucket) = BucketIndex::from_flat_index(idx as usize);
        SlotIndex { bucket_idx, index_in_bucket }
    }

    // SAFETY: Buckets must be managed solely by functions here (i.e., get/put on SlotIndex) and
    // `self` comes from SlotIndex::from_index
    #[inline]
    unsafe fn get<'a, V: Copy>(
        &self,
        buckets: &'a [AtomicPtr<CacheEntry<V>>; 21],
    ) -> Option<&'a CacheEntry<V>> {
        let bucket = &buckets[self.bucket_idx];
        let ptr = bucket.load(Ordering::Acquire);
        // Bucket is not yet initialized: then we obviously won't find this entry in that bucket.
        if ptr.is_null() {
            return None;
        }
        debug_assert!(self.index_in_bucket < self.bucket_idx.capacity());
        // SAFETY: `bucket` was allocated (so <= isize in total bytes) to hold `entries`, so this
        // must be inbounds.
        let entry_ptr = unsafe { ptr.add(self.index_in_bucket) };

        // SAFETY: initialized bucket has zeroed all memory within the bucket, so we are valid for
        // CacheEntry access.
        Some(unsafe { &*entry_ptr })
    }

    fn bucket_ptr<V>(&self, bucket: &AtomicPtr<CacheEntry<V>>) -> *mut CacheEntry<V> {
        let ptr = bucket.load(Ordering::Acquire);
        if ptr.is_null() { Self::initialize_bucket(bucket, self.bucket_idx) } else { ptr }
    }

    #[cold]
    #[inline(never)]
    fn initialize_bucket<V>(
        bucket: &AtomicPtr<CacheEntry<V>>,
        bucket_idx: BucketIndex,
    ) -> *mut CacheEntry<V> {
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
                std::alloc::Layout::array::<CacheEntry<V>>(bucket_idx.capacity()).unwrap();
            // This is more of a sanity check -- this code is very cold, so it's safe to pay a
            // little extra cost here.
            assert!(bucket_layout.size() > 0);
            // SAFETY: Just checked that size is non-zero.
            let allocated =
                unsafe { std::alloc::alloc_zeroed(bucket_layout).cast::<CacheEntry<V>>() };
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
    fn get_or_init<'a, V>(&self, buckets: &'a [AtomicPtr<CacheEntry<V>>; 21]) -> &'a CacheEntry<V> {
        let bucket = &buckets[self.bucket_idx];
        let ptr = self.bucket_ptr(bucket);

        debug_assert!(self.index_in_bucket < self.bucket_idx.capacity());
        // SAFETY: `bucket` was allocated (so <= isize in total bytes) to hold `entries`, so this
        // must be inbounds.
        let entry_ptr = unsafe { ptr.add(self.index_in_bucket) };

        // SAFETY: initialized bucket has zeroed all memory within the bucket, so we are valid for
        // CacheEntry access.
        unsafe { &*entry_ptr }
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
    // The total number of entries if all buckets are initialized is 2^32.
    buckets: [AtomicPtr<CacheEntry<V>>; BUCKETS],

    key: PhantomData<(K, I)>,
}

impl<K: Idx, V, I> Default for VecCache<K, V, I> {
    fn default() -> Self {
        VecCache { buckets: Default::default(), key: PhantomData }
    }
}

// SAFETY: No access to `V` is made.
unsafe impl<K: Idx, #[may_dangle] V, I> Drop for VecCache<K, V, I> {
    fn drop(&mut self) {
        // We have unique ownership, so no locks etc. are needed. Since `K` is `Copy`,
        // we are also guaranteed to just need to deallocate any large arrays (not iterate over
        // contents).
        assert!(!std::mem::needs_drop::<K>());

        for (idx, bucket) in BucketIndex::enumerate_buckets(&self.buckets) {
            let bucket = bucket.load(Ordering::Acquire);
            if !bucket.is_null() {
                if std::mem::needs_drop::<V>() {
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(bucket, ENTRIES_BY_BUCKET[idx]))
                    }
                }
                let layout =
                    std::alloc::Layout::array::<CacheEntry<V>>(ENTRIES_BY_BUCKET[idx]).unwrap();
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
    I: Idx + Copy,
{
    #[inline]
    pub fn lookup(&self, key: K) -> &CacheEntry<V> {
        let key = u32::try_from(key.index()).unwrap();
        let slot_idx = SlotIndex::from_index(key);
        slot_idx.get_or_init(&self.buckets)
    }

    pub fn for_each(&self, mut f: impl FnMut(K, &V, I)) {
        for (bucket_idx, bucket) in BucketIndex::enumerate_buckets(&self.buckets) {
            let mut idx =
                if let BucketIndex::Bucket00 = bucket_idx { 0 } else { bucket_idx.capacity() };
            let bucket = bucket.load(Ordering::Acquire);
            if !bucket.is_null() {
                let entries =
                    unsafe { slice::from_raw_parts(bucket, ENTRIES_BY_BUCKET[bucket_idx]) };
                for entry in entries {
                    if let Some((value, additional_index)) = entry.get_finished() {
                        let key = K::new(idx);
                        f(key, value, I::new(additional_index as usize));
                    }
                    idx += 1;
                }
            }
        }
    }
}

/// Index into an array of buckets.
///
/// Using an enum lets us tell the compiler that values range from 0 to 20,
/// allowing array bounds checks to be optimized away,
/// without having to resort to pattern types or other unstable features.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum BucketIndex {
    // tidy-alphabetical-start
    Bucket00,
    Bucket01,
    Bucket02,
    Bucket03,
    Bucket04,
    Bucket05,
    Bucket06,
    Bucket07,
    Bucket08,
    Bucket09,
    Bucket10,
    Bucket11,
    Bucket12,
    Bucket13,
    Bucket14,
    Bucket15,
    Bucket16,
    Bucket17,
    Bucket18,
    Bucket19,
    Bucket20,
    // tidy-alphabetical-end
}

impl Debug for BucketIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.to_usize(), f)
    }
}

impl BucketIndex {
    /// Capacity of bucket 0 (and also of bucket 1).
    const BUCKET_0_CAPACITY: usize = 1 << (Self::NONZERO_BUCKET_SHIFT_ADJUST + 1);
    /// Adjustment factor from the highest-set-bit-position of a flat index,
    /// to its corresponding bucket number.
    ///
    /// For example, the first flat-index in bucket 2 is 8192.
    /// Its highest-set-bit-position is `(8192).ilog2() == 13`, and subtracting
    /// the adjustment factor of 11 gives the bucket number of 2.
    const NONZERO_BUCKET_SHIFT_ADJUST: usize = 11;

    #[inline(always)]
    const fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    const fn from_raw(raw: usize) -> Self {
        match raw {
            // tidy-alphabetical-start
            00 => Self::Bucket00,
            01 => Self::Bucket01,
            02 => Self::Bucket02,
            03 => Self::Bucket03,
            04 => Self::Bucket04,
            05 => Self::Bucket05,
            06 => Self::Bucket06,
            07 => Self::Bucket07,
            08 => Self::Bucket08,
            09 => Self::Bucket09,
            10 => Self::Bucket10,
            11 => Self::Bucket11,
            12 => Self::Bucket12,
            13 => Self::Bucket13,
            14 => Self::Bucket14,
            15 => Self::Bucket15,
            16 => Self::Bucket16,
            17 => Self::Bucket17,
            18 => Self::Bucket18,
            19 => Self::Bucket19,
            20 => Self::Bucket20,
            // tidy-alphabetical-end
            _ => panic!("bucket index out of range"),
        }
    }

    /// Total number of slots in this bucket.
    #[inline(always)]
    const fn capacity(self) -> usize {
        match self {
            Self::Bucket00 => Self::BUCKET_0_CAPACITY,
            // Bucket 1 has a capacity of `1 << (1 + 11) == pow(2, 12) == 4096`.
            // Bucket 2 has a capacity of `1 << (2 + 11) == pow(2, 13) == 8192`.
            _ => 1 << (self.to_usize() + Self::NONZERO_BUCKET_SHIFT_ADJUST),
        }
    }

    /// Converts a flat index in the range `0..=u32::MAX` into a bucket index,
    /// and a slot offset within that bucket.
    ///
    /// Panics if `flat > u32::MAX`.
    #[inline(always)]
    const fn from_flat_index(flat: usize) -> (Self, usize) {
        if flat > u32::MAX as usize {
            panic!();
        }

        // If the index is in bucket 0, the conversion is trivial.
        // This also avoids calling `ilog2` when `flat == 0`.
        if flat < Self::BUCKET_0_CAPACITY {
            return (Self::Bucket00, flat);
        }

        // General-case conversion for a non-zero bucket index.
        //
        //              | bucket |   slot
        // flat | ilog2 |  index | offset
        // ------------------------------
        // 4096 |    12 |      1 |      0
        // 4097 |    12 |      1 |      1
        // ...
        // 8191 |    12 |      1 |   4095
        // 8192 |    13 |      2 |      0
        let highest_bit_pos = flat.ilog2() as usize;
        let bucket_index =
            BucketIndex::from_raw(highest_bit_pos - Self::NONZERO_BUCKET_SHIFT_ADJUST);

        // Clear the highest-set bit (which selects the bucket) to get the
        // slot offset within this bucket.
        let slot_offset = flat - (1 << highest_bit_pos);

        (bucket_index, slot_offset)
    }

    #[inline(always)]
    fn iter_all() -> impl ExactSizeIterator<Item = Self> {
        (0usize..BUCKETS).map(BucketIndex::from_raw)
    }

    #[inline(always)]
    fn enumerate_buckets<T>(buckets: &[T; BUCKETS]) -> impl ExactSizeIterator<Item = (Self, &T)> {
        BucketIndex::iter_all().zip(buckets)
    }
}

impl<T> Index<BucketIndex> for [T; BUCKETS] {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: BucketIndex) -> &Self::Output {
        // The optimizer should be able to see that see that a bucket index is
        // always in-bounds, and omit the runtime bounds check.
        &self[index.to_usize()]
    }
}

impl<T> IndexMut<BucketIndex> for [T; BUCKETS] {
    #[inline(always)]
    fn index_mut(&mut self, index: BucketIndex) -> &mut Self::Output {
        &mut self[index.to_usize()]
    }
}
