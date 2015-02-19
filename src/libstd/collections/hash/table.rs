// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15883

use self::BucketState::*;

use clone::Clone;
use cmp;
use hash::{Hash, Hasher};
use iter::{Iterator, IteratorExt, ExactSizeIterator, count};
use marker::{Copy, Send, Sync, Sized, self};
use mem::{min_align_of, size_of};
use mem;
use num::{Int, UnsignedInt};
use ops::{Deref, DerefMut, Drop};
use option::Option;
use option::Option::{Some, None};
use ptr::{self, PtrExt, copy_nonoverlapping_memory, Unique, zero_memory};
use rt::heap::{allocate, deallocate, EMPTY};
use collections::hash_state::HashState;

const EMPTY_BUCKET: u64 = 0u64;

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, keys, and values.
///
/// This design uses less memory and is a lot faster than the naive
/// `Vec<Option<u64, K, V>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Essential invariants of this structure:
///
///   - if t.hashes[i] == EMPTY_BUCKET, then `Bucket::at_index(&t, i).raw`
///     points to 'undefined' contents. Don't read from it. This invariant is
///     enforced outside this module with the `EmptyBucket`, `FullBucket`,
///     and `SafeHash` types.
///
///   - An `EmptyBucket` is only constructed at an index with
///     a hash of EMPTY_BUCKET.
///
///   - A `FullBucket` is only constructed at an index with a
///     non-EMPTY_BUCKET hash.
///
///   - A `SafeHash` is only constructed for non-`EMPTY_BUCKET` hash. We get
///     around hashes of zero by changing them to 0x8000_0000_0000_0000,
///     which will likely map to the same bucket, while not being confused
///     with "empty".
///
///   - All three "arrays represented by pointers" are the same length:
///     `capacity`. This is set at creation and never changes. The arrays
///     are unzipped to save space (we don't have to pay for the padding
///     between odd sized elements, such as in a map from u64 to u8), and
///     be more cache aware (scanning through 8 hashes brings in at most
///     2 cache lines, since they're all right beside each other).
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<u64, K, V>>`.
#[unsafe_no_drop_flag]
pub struct RawTable<K, V> {
    capacity: usize,
    size:     usize,
    hashes:   Unique<u64>,

    // Because K/V do not appear directly in any of the types in the struct,
    // inform rustc that in fact instances of K and V are reachable from here.
    marker:   marker::PhantomData<(K,V)>,
}

unsafe impl<K: Send, V: Send> Send for RawTable<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RawTable<K, V> {}

struct RawBucket<K, V> {
    hash: *mut u64,
    key:  *mut K,
    val:  *mut V,
    _marker: marker::PhantomData<(K,V)>,
}

impl<K,V> Copy for RawBucket<K,V> {}

pub struct Bucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   usize,
    table: M
}

impl<K,V,M:Copy> Copy for Bucket<K,V,M> {}

pub struct EmptyBucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   usize,
    table: M
}

pub struct FullBucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   usize,
    table: M
}

pub type EmptyBucketImm<'table, K, V> = EmptyBucket<K, V, &'table RawTable<K, V>>;
pub type  FullBucketImm<'table, K, V> =  FullBucket<K, V, &'table RawTable<K, V>>;

pub type EmptyBucketMut<'table, K, V> = EmptyBucket<K, V, &'table mut RawTable<K, V>>;
pub type  FullBucketMut<'table, K, V> =  FullBucket<K, V, &'table mut RawTable<K, V>>;

pub enum BucketState<K, V, M> {
    Empty(EmptyBucket<K, V, M>),
    Full(FullBucket<K, V, M>),
}

// A GapThenFull encapsulates the state of two consecutive buckets at once.
// The first bucket, called the gap, is known to be empty.
// The second bucket is full.
struct GapThenFull<K, V, M> {
    gap: EmptyBucket<K, V, ()>,
    full: FullBucket<K, V, M>,
}

/// A hash that is not zero, since we use a hash of zero to represent empty
/// buckets.
#[derive(PartialEq, Copy)]
pub struct SafeHash {
    hash: u64,
}

impl SafeHash {
    /// Peek at the hash value, which is guaranteed to be non-zero.
    #[inline(always)]
    pub fn inspect(&self) -> u64 { self.hash }
}

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
#[cfg(stage0)]
pub fn make_hash<T: ?Sized, S, H>(hash_state: &S, t: &T) -> SafeHash
    where T: Hash<H>,
          S: HashState<Hasher=H>,
          H: Hasher<Output=u64>
{
    let mut state = hash_state.hasher();
    t.hash(&mut state);
    // We need to avoid 0u64 in order to prevent collisions with
    // EMPTY_HASH. We can maintain our precious uniform distribution
    // of initial indexes by unconditionally setting the MSB,
    // effectively reducing 64-bits hashes to 63 bits.
    SafeHash { hash: 0x8000_0000_0000_0000 | state.finish() }
}

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
#[cfg(not(stage0))]
pub fn make_hash<T: ?Sized, S>(hash_state: &S, t: &T) -> SafeHash
    where T: Hash, S: HashState
{
    let mut state = hash_state.hasher();
    t.hash(&mut state);
    // We need to avoid 0u64 in order to prevent collisions with
    // EMPTY_HASH. We can maintain our precious uniform distribution
    // of initial indexes by unconditionally setting the MSB,
    // effectively reducing 64-bits hashes to 63 bits.
    SafeHash { hash: 0x8000_0000_0000_0000 | state.finish() }
}

// `replace` casts a `*u64` to a `*SafeHash`. Since we statically
// ensure that a `FullBucket` points to an index with a non-zero hash,
// and a `SafeHash` is just a `u64` with a different name, this is
// safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `u64`. If you need to change the size of `SafeHash` (and
// consequently made this test fail), `replace` needs to be
// modified to no longer assume this.
#[test]
fn can_alias_safehash_as_u64() {
    assert_eq!(size_of::<SafeHash>(), size_of::<u64>())
}

impl<K, V> RawBucket<K, V> {
    unsafe fn offset(self, count: isize) -> RawBucket<K, V> {
        RawBucket {
            hash: self.hash.offset(count),
            key:  self.key.offset(count),
            val:  self.val.offset(count),
            _marker: marker::PhantomData,
        }
    }
}

// Buckets hold references to the table.
impl<K, V, M> FullBucket<K, V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl<K, V, M> EmptyBucket<K, V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
}

impl<K, V, M> Bucket<K, V, M> {
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl<K, V, M: Deref<Target=RawTable<K, V>>> Bucket<K, V, M> {
    pub fn new(table: M, hash: SafeHash) -> Bucket<K, V, M> {
        Bucket::at_index(table, hash.inspect() as usize)
    }

    pub fn at_index(table: M, ib_index: usize) -> Bucket<K, V, M> {
        let ib_index = ib_index & (table.capacity() - 1);
        Bucket {
            raw: unsafe {
               table.first_bucket_raw().offset(ib_index as isize)
            },
            idx: ib_index,
            table: table
        }
    }

    pub fn first(table: M) -> Bucket<K, V, M> {
        Bucket {
            raw: table.first_bucket_raw(),
            idx: 0,
            table: table
        }
    }

    /// Reads a bucket at a given index, returning an enum indicating whether
    /// it's initialized or not. You need to match on this enum to get
    /// the appropriate types to call most of the other functions in
    /// this module.
    pub fn peek(self) -> BucketState<K, V, M> {
        match unsafe { *self.raw.hash } {
            EMPTY_BUCKET =>
                Empty(EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                }),
            _ =>
                Full(FullBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                })
        }
    }

    /// Modifies the bucket pointer in place to make it point to the next slot.
    pub fn next(&mut self) {
        // Branchless bucket iteration step.
        // As we reach the end of the table...
        // We take the current idx:          0111111b
        // Xor it by its increment:        ^ 1000000b
        //                               ------------
        //                                   1111111b
        // Then AND with the capacity:     & 1000000b
        //                               ------------
        // to get the backwards offset:      1000000b
        // ... and it's zero at all other times.
        let maybe_wraparound_dist = (self.idx ^ (self.idx + 1)) & self.table.capacity();
        // Finally, we obtain the offset 1 or the offset -cap + 1.
        let dist = 1 - (maybe_wraparound_dist as isize);

        self.idx += 1;

        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }
}

impl<K, V, M: Deref<Target=RawTable<K, V>>> EmptyBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table
        }
    }

    pub fn gap_peek(self) -> Option<GapThenFull<K, V, M>> {
        let gap = EmptyBucket {
            raw: self.raw,
            idx: self.idx,
            table: ()
        };

        match self.next().peek() {
            Full(bucket) => {
                Some(GapThenFull {
                    gap: gap,
                    full: bucket
                })
            }
            Empty(..) => None
        }
    }
}

impl<K, V, M: Deref<Target=RawTable<K, V>> + DerefMut> EmptyBucket<K, V, M> {
    /// Puts given key and value pair, along with the key's hash,
    /// into this bucket in the hashtable. Note how `self` is 'moved' into
    /// this function, because this slot will no longer be empty when
    /// we return! A `FullBucket` is returned for later use, pointing to
    /// the newly-filled slot in the hashtable.
    ///
    /// Use `make_hash` to construct a `SafeHash` to pass to this function.
    pub fn put(mut self, hash: SafeHash, key: K, value: V)
               -> FullBucket<K, V, M> {
        unsafe {
            *self.raw.hash = hash.inspect();
            ptr::write(self.raw.key, key);
            ptr::write(self.raw.val, value);
        }

        self.table.size += 1;

        FullBucket { raw: self.raw, idx: self.idx, table: self.table }
    }
}

impl<K, V, M: Deref<Target=RawTable<K, V>>> FullBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table
        }
    }

    /// Get the distance between this bucket and the 'ideal' location
    /// as determined by the key's hash stored in it.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// initial bucket", or DIB. Also known as "probe count".
    pub fn distance(&self) -> usize {
        // Calculates the distance one has to travel when going from
        // `hash mod capacity` onwards to `idx mod capacity`, wrapping around
        // if the destination is not reached before the end of the table.
        (self.idx - self.hash().inspect() as usize) & (self.table.capacity() - 1)
    }

    #[inline]
    pub fn hash(&self) -> SafeHash {
        unsafe {
            SafeHash {
                hash: *self.raw.hash
            }
        }
    }

    /// Gets references to the key and value at a given index.
    pub fn read(&self) -> (&K, &V) {
        unsafe {
            (&*self.raw.key,
             &*self.raw.val)
        }
    }
}

impl<K, V, M: Deref<Target=RawTable<K, V>> + DerefMut> FullBucket<K, V, M> {
    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(mut self) -> (EmptyBucket<K, V, M>, K, V) {
        self.table.size -= 1;

        unsafe {
            *self.raw.hash = EMPTY_BUCKET;
            (
                EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                },
                ptr::read(self.raw.key),
                ptr::read(self.raw.val)
            )
        }
    }

    pub fn replace(&mut self, h: SafeHash, k: K, v: V) -> (SafeHash, K, V) {
        unsafe {
            let old_hash = ptr::replace(self.raw.hash as *mut SafeHash, h);
            let old_key  = ptr::replace(self.raw.key,  k);
            let old_val  = ptr::replace(self.raw.val,  v);

            (old_hash, old_key, old_val)
        }
    }

    /// Gets mutable references to the key and value at a given index.
    pub fn read_mut(&mut self) -> (&mut K, &mut V) {
        unsafe {
            (&mut *self.raw.key,
             &mut *self.raw.val)
        }
    }
}

impl<'t, K, V, M: Deref<Target=RawTable<K, V>> + 't> FullBucket<K, V, M> {
    /// Exchange a bucket state for immutable references into the table.
    /// Because the underlying reference to the table is also consumed,
    /// no further changes to the structure of the table are possible;
    /// in exchange for this, the returned references have a longer lifetime
    /// than the references returned by `read()`.
    pub fn into_refs(self) -> (&'t K, &'t V) {
        unsafe {
            (&*self.raw.key,
             &*self.raw.val)
        }
    }
}

impl<'t, K, V, M: Deref<Target=RawTable<K, V>> + DerefMut + 't> FullBucket<K, V, M> {
    /// This works similarly to `into_refs`, exchanging a bucket state
    /// for mutable references into the table.
    pub fn into_mut_refs(self) -> (&'t mut K, &'t mut V) {
        unsafe {
            (&mut *self.raw.key,
             &mut *self.raw.val)
        }
    }
}

impl<K, V, M> BucketState<K, V, M> {
    // For convenience.
    pub fn expect_full(self) -> FullBucket<K, V, M> {
        match self {
            Full(full) => full,
            Empty(..) => panic!("Expected full bucket")
        }
    }
}

impl<K, V, M: Deref<Target=RawTable<K, V>>> GapThenFull<K, V, M> {
    #[inline]
    pub fn full(&self) -> &FullBucket<K, V, M> {
        &self.full
    }

    pub fn shift(mut self) -> Option<GapThenFull<K, V, M>> {
        unsafe {
            *self.gap.raw.hash = mem::replace(&mut *self.full.raw.hash, EMPTY_BUCKET);
            copy_nonoverlapping_memory(self.gap.raw.key, self.full.raw.key, 1);
            copy_nonoverlapping_memory(self.gap.raw.val, self.full.raw.val, 1);
        }

        let FullBucket { raw: prev_raw, idx: prev_idx, .. } = self.full;

        match self.full.next().peek() {
            Full(bucket) => {
                self.gap.raw = prev_raw;
                self.gap.idx = prev_idx;

                self.full = bucket;

                Some(self)
            }
            Empty(..) => None
        }
    }
}


/// Rounds up to a multiple of a power of two. Returns the closest multiple
/// of `target_alignment` that is higher or equal to `unrounded`.
///
/// # Panics
///
/// Panics if `target_alignment` is not a power of two.
fn round_up_to_next(unrounded: usize, target_alignment: usize) -> usize {
    assert!(target_alignment.is_power_of_two());
    (unrounded + target_alignment - 1) & !(target_alignment - 1)
}

#[test]
fn test_rounding() {
    assert_eq!(round_up_to_next(0, 4), 0);
    assert_eq!(round_up_to_next(1, 4), 4);
    assert_eq!(round_up_to_next(2, 4), 4);
    assert_eq!(round_up_to_next(3, 4), 4);
    assert_eq!(round_up_to_next(4, 4), 4);
    assert_eq!(round_up_to_next(5, 4), 8);
}

// Returns a tuple of (key_offset, val_offset),
// from the start of a mallocated array.
fn calculate_offsets(hashes_size: usize,
                     keys_size: usize, keys_align: usize,
                     vals_align: usize)
                     -> (usize, usize) {
    let keys_offset = round_up_to_next(hashes_size, keys_align);
    let end_of_keys = keys_offset + keys_size;

    let vals_offset = round_up_to_next(end_of_keys, vals_align);

    (keys_offset, vals_offset)
}

// Returns a tuple of (minimum required malloc alignment, hash_offset,
// array_size), from the start of a mallocated array.
fn calculate_allocation(hash_size: usize, hash_align: usize,
                        keys_size: usize, keys_align: usize,
                        vals_size: usize, vals_align: usize)
                        -> (usize, usize, usize) {
    let hash_offset = 0;
    let (_, vals_offset) = calculate_offsets(hash_size,
                                             keys_size, keys_align,
                                                        vals_align);
    let end_of_vals = vals_offset + vals_size;

    let min_align = cmp::max(hash_align, cmp::max(keys_align, vals_align));

    (min_align, hash_offset, end_of_vals)
}

#[test]
fn test_offset_calculation() {
    assert_eq!(calculate_allocation(128, 8, 15, 1, 4,  4), (8, 0, 148));
    assert_eq!(calculate_allocation(3,   1, 2,  1, 1,  1), (1, 0, 6));
    assert_eq!(calculate_allocation(6,   2, 12, 4, 24, 8), (8, 0, 48));
    assert_eq!(calculate_offsets(128, 15, 1, 4), (128, 144));
    assert_eq!(calculate_offsets(3,   2,  1, 1), (3,   5));
    assert_eq!(calculate_offsets(6,   12, 4, 8), (8,   24));
}

impl<K, V> RawTable<K, V> {
    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    unsafe fn new_uninitialized(capacity: usize) -> RawTable<K, V> {
        if capacity == 0 {
            return RawTable {
                size: 0,
                capacity: 0,
                hashes: Unique::new(EMPTY as *mut u64),
                marker: marker::PhantomData,
            };
        }

        // No need for `checked_mul` before a more restrictive check performed
        // later in this method.
        let hashes_size = capacity * size_of::<u64>();
        let keys_size   = capacity * size_of::< K >();
        let vals_size   = capacity * size_of::< V >();

        // Allocating hashmaps is a little tricky. We need to allocate three
        // arrays, but since we know their sizes and alignments up front,
        // we just allocate a single array, and then have the subarrays
        // point into it.
        //
        // This is great in theory, but in practice getting the alignment
        // right is a little subtle. Therefore, calculating offsets has been
        // factored out into a different function.
        let (malloc_alignment, hash_offset, size) =
            calculate_allocation(
                hashes_size, min_align_of::<u64>(),
                keys_size,   min_align_of::< K >(),
                vals_size,   min_align_of::< V >());

        // One check for overflow that covers calculation and rounding of size.
        let size_of_bucket = size_of::<u64>().checked_add(size_of::<K>()).unwrap()
                                             .checked_add(size_of::<V>()).unwrap();
        assert!(size >= capacity.checked_mul(size_of_bucket)
                                .expect("capacity overflow"),
                "capacity overflow");

        let buffer = allocate(size, malloc_alignment);
        if buffer.is_null() { ::alloc::oom() }

        let hashes = buffer.offset(hash_offset as isize) as *mut u64;

        RawTable {
            capacity: capacity,
            size:     0,
            hashes:   Unique::new(hashes),
            marker:   marker::PhantomData,
        }
    }

    fn first_bucket_raw(&self) -> RawBucket<K, V> {
        let hashes_size = self.capacity * size_of::<u64>();
        let keys_size = self.capacity * size_of::<K>();

        let buffer = *self.hashes as *mut u8;
        let (keys_offset, vals_offset) = calculate_offsets(hashes_size,
                                                           keys_size, min_align_of::<K>(),
                                                           min_align_of::<V>());

        unsafe {
            RawBucket {
                hash: *self.hashes,
                key:  buffer.offset(keys_offset as isize) as *mut K,
                val:  buffer.offset(vals_offset as isize) as *mut V,
                _marker: marker::PhantomData,
            }
        }
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    pub fn new(capacity: usize) -> RawTable<K, V> {
        unsafe {
            let ret = RawTable::new_uninitialized(capacity);
            zero_memory(*ret.hashes, capacity);
            ret
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> usize {
        self.size
    }

    fn raw_buckets(&self) -> RawBuckets<K, V> {
        RawBuckets {
            raw: self.first_bucket_raw(),
            hashes_end: unsafe {
                self.hashes.offset(self.capacity as isize)
            },
            marker: marker::PhantomData,
        }
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.raw_buckets(),
            elems_left: self.size(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            iter: self.raw_buckets(),
            elems_left: self.size(),
        }
    }

    pub fn into_iter(self) -> IntoIter<K, V> {
        let RawBuckets { raw, hashes_end, .. } = self.raw_buckets();
        // Replace the marker regardless of lifetime bounds on parameters.
        IntoIter {
            iter: RawBuckets {
                raw: raw,
                hashes_end: hashes_end,
                marker: marker::PhantomData,
            },
            table: self,
        }
    }

    pub fn drain(&mut self) -> Drain<K, V> {
        let RawBuckets { raw, hashes_end, .. } = self.raw_buckets();
        // Replace the marker regardless of lifetime bounds on parameters.
        Drain {
            iter: RawBuckets {
                raw: raw,
                hashes_end: hashes_end,
                marker: marker::PhantomData,
            },
            table: self,
        }
    }

    /// Returns an iterator that copies out each entry. Used while the table
    /// is being dropped.
    unsafe fn rev_move_buckets(&mut self) -> RevMoveBuckets<K, V> {
        let raw_bucket = self.first_bucket_raw();
        RevMoveBuckets {
            raw: raw_bucket.offset(self.capacity as isize),
            hashes_end: raw_bucket.hash,
            elems_left: self.size,
            marker:     marker::PhantomData,
        }
    }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
struct RawBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64,

    // Strictly speaking, this should be &'a (K,V), but that would
    // require that K:'a, and we often use RawBuckets<'static...> for
    // move iterations, so that messes up a lot of other things. So
    // just use `&'a (K,V)` as this is not a publicly exposed type
    // anyway.
    marker: marker::PhantomData<&'a ()>,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, K, V> Clone for RawBuckets<'a, K, V> {
    fn clone(&self) -> RawBuckets<'a, K, V> {
        RawBuckets {
            raw: self.raw,
            hashes_end: self.hashes_end,
            marker: marker::PhantomData,
        }
    }
}


impl<'a, K, V> Iterator for RawBuckets<'a, K, V> {
    type Item = RawBucket<K, V>;

    fn next(&mut self) -> Option<RawBucket<K, V>> {
        while self.raw.hash != self.hashes_end {
            unsafe {
                // We are swapping out the pointer to a bucket and replacing
                // it with the pointer to the next one.
                let prev = ptr::replace(&mut self.raw, self.raw.offset(1));
                if *prev.hash != EMPTY_BUCKET {
                    return Some(prev);
                }
            }
        }

        None
    }
}

/// An iterator that moves out buckets in reverse order. It leaves the table
/// in an inconsistent state and should only be used for dropping
/// the table's remaining entries. It's used in the implementation of Drop.
struct RevMoveBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64,
    elems_left: usize,

    // As above, `&'a (K,V)` would seem better, but we often use
    // 'static for the lifetime, and this is not a publicly exposed
    // type.
    marker: marker::PhantomData<&'a ()>,
}

impl<'a, K, V> Iterator for RevMoveBuckets<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if self.elems_left == 0 {
            return None;
        }

        loop {
            debug_assert!(self.raw.hash != self.hashes_end);

            unsafe {
                self.raw = self.raw.offset(-1);

                if *self.raw.hash != EMPTY_BUCKET {
                    self.elems_left -= 1;
                    return Some((
                        ptr::read(self.raw.key),
                        ptr::read(self.raw.val)
                    ));
                }
            }
        }
    }
}

/// Iterator over shared references to entries in a table.
pub struct Iter<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: usize,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, K, V> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            iter: self.iter.clone(),
            elems_left: self.elems_left
        }
    }
}


/// Iterator over mutable references to entries in a table.
pub struct IterMut<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: usize,
}

/// Iterator over the entries in a table, consuming the table.
pub struct IntoIter<K, V> {
    table: RawTable<K, V>,
    iter: RawBuckets<'static, K, V>
}

/// Iterator over the entries in a table, clearing the table.
pub struct Drain<'a, K: 'a, V: 'a> {
    table: &'a mut RawTable<K, V>,
    iter: RawBuckets<'static, K, V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                (&*bucket.key,
                 &*bucket.val)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize { self.elems_left }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                (&*bucket.key,
                 &mut *bucket.val)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize { self.elems_left }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (SafeHash, K, V);

    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                (
                    SafeHash {
                        hash: *bucket.hash,
                    },
                    ptr::read(bucket.key),
                    ptr::read(bucket.val)
                )
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.table.size();
        (size, Some(size))
    }
}
impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize { self.table.size() }
}

impl<'a, K, V> Iterator for Drain<'a, K, V> {
    type Item = (SafeHash, K, V);

    #[inline]
    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                (
                    SafeHash {
                        hash: ptr::replace(bucket.hash, EMPTY_BUCKET),
                    },
                    ptr::read(bucket.key),
                    ptr::read(bucket.val)
                )
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.table.size();
        (size, Some(size))
    }
}
impl<'a, K, V> ExactSizeIterator for Drain<'a, K, V> {
    fn len(&self) -> usize { self.table.size() }
}

#[unsafe_destructor]
impl<'a, K: 'a, V: 'a> Drop for Drain<'a, K, V> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}
    }
}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> RawTable<K, V> {
        unsafe {
            let mut new_ht = RawTable::new_uninitialized(self.capacity());

            {
                let cap = self.capacity();
                let mut new_buckets = Bucket::first(&mut new_ht);
                let mut buckets = Bucket::first(self);
                while buckets.index() != cap {
                    match buckets.peek() {
                        Full(full) => {
                            let (h, k, v) = {
                                let (k, v) = full.read();
                                (full.hash(), k.clone(), v.clone())
                            };
                            *new_buckets.raw.hash = h.inspect();
                            ptr::write(new_buckets.raw.key, k);
                            ptr::write(new_buckets.raw.val, v);
                        }
                        Empty(..) => {
                            *new_buckets.raw.hash = EMPTY_BUCKET;
                        }
                    }
                    new_buckets.next();
                    buckets.next();
                }
            };

            new_ht.size = self.size();

            new_ht
        }
    }
}

#[unsafe_destructor]
impl<K, V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }

        // This is done in reverse because we've likely partially taken
        // some elements out with `.into_iter()` from the front.
        // Check if the size is 0, so we don't do a useless scan when
        // dropping empty tables such as on resize.
        // Also avoid double drop of elements that have been already moved out.
        unsafe {
            for _ in self.rev_move_buckets() {}
        }

        let hashes_size = self.capacity * size_of::<u64>();
        let keys_size = self.capacity * size_of::<K>();
        let vals_size = self.capacity * size_of::<V>();
        let (align, _, size) = calculate_allocation(hashes_size, min_align_of::<u64>(),
                                                    keys_size, min_align_of::<K>(),
                                                    vals_size, min_align_of::<V>());

        unsafe {
            deallocate(*self.hashes as *mut u8, size, align);
            // Remember how everything was allocated out of one buffer
            // during initialization? We only need one call to free here.
        }
    }
}
