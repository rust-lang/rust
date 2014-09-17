// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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

use clone::Clone;
use cmp;
use hash::{Hash, Hasher};
use iter::{Iterator, count};
use kinds::marker;
use mem::{min_align_of, size_of};
use mem;
use num::{CheckedAdd, CheckedMul, is_power_of_two};
use ops::{Deref, DerefMut, Drop};
use option::{Some, None, Option};
use ptr::{RawPtr, copy_nonoverlapping_memory, zero_memory};
use ptr;
use rt::heap::{allocate, deallocate};

static EMPTY_BUCKET: u64 = 0u64;

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
    capacity: uint,
    size:     uint,
    hashes:   *mut u64,
    // Because K/V do not appear directly in any of the types in the struct,
    // inform rustc that in fact instances of K and V are reachable from here.
    marker:   marker::CovariantType<(K,V)>,
}

struct RawBucket<K, V> {
    hash: *mut u64,
    key:  *mut K,
    val:  *mut V
}

pub struct Bucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   uint,
    table: M
}

pub struct EmptyBucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   uint,
    table: M
}

pub struct FullBucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   uint,
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
#[deriving(PartialEq)]
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
pub fn make_hash<T: Hash<S>, S, H: Hasher<S>>(hasher: &H, t: &T) -> SafeHash {
    match hasher.hash(t) {
        // This constant is exceedingly likely to hash to the same
        // bucket, but it won't be counted as empty! Just so we can maintain
        // our precious uniform distribution of initial indexes.
        EMPTY_BUCKET => SafeHash { hash: 0x8000_0000_0000_0000 },
        h            => SafeHash { hash: h },
    }
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
    unsafe fn offset(self, count: int) -> RawBucket<K, V> {
        RawBucket {
            hash: self.hash.offset(count),
            key:  self.key.offset(count),
            val:  self.val.offset(count),
        }
    }
}

// For parameterizing over mutability.
impl<'t, K, V> Deref<RawTable<K, V>> for &'t RawTable<K, V> {
    fn deref(&self) -> &RawTable<K, V> {
        &**self
    }
}

impl<'t, K, V> Deref<RawTable<K, V>> for &'t mut RawTable<K, V> {
    fn deref(&self) -> &RawTable<K,V> {
        &**self
    }
}

impl<'t, K, V> DerefMut<RawTable<K, V>> for &'t mut RawTable<K, V> {
    fn deref_mut(&mut self) -> &mut RawTable<K,V> {
        &mut **self
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
    pub fn index(&self) -> uint {
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
    pub fn index(&self) -> uint {
        self.idx
    }
}

impl<K, V, M: Deref<RawTable<K, V>>> Bucket<K, V, M> {
    pub fn new(table: M, hash: &SafeHash) -> Bucket<K, V, M> {
        Bucket::at_index(table, hash.inspect() as uint)
    }

    pub fn at_index(table: M, ib_index: uint) -> Bucket<K, V, M> {
        let ib_index = ib_index & (table.capacity() - 1);
        Bucket {
            raw: unsafe {
               table.first_bucket_raw().offset(ib_index as int)
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
        let dist = 1i - (maybe_wraparound_dist as int);

        self.idx += 1;

        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }
}

impl<K, V, M: Deref<RawTable<K, V>>> EmptyBucket<K, V, M> {
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

impl<K, V, M: DerefMut<RawTable<K, V>>> EmptyBucket<K, V, M> {
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

impl<K, V, M: Deref<RawTable<K, V>>> FullBucket<K, V, M> {
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
    pub fn distance(&self) -> uint {
        // Calculates the distance one has to travel when going from
        // `hash mod capacity` onwards to `idx mod capacity`, wrapping around
        // if the destination is not reached before the end of the table.
        (self.idx - self.hash().inspect() as uint) & (self.table.capacity() - 1)
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

impl<K, V, M: DerefMut<RawTable<K, V>>> FullBucket<K, V, M> {
    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(mut self) -> (EmptyBucket<K, V, M>, K, V) {
        let key = self.raw.key as *const K;
        let val = self.raw.val as *const V;

        self.table.size -= 1;

        unsafe {
            *self.raw.hash = EMPTY_BUCKET;
            (
                EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                },
                ptr::read(key),
                ptr::read(val)
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

impl<'t, K, V, M: Deref<RawTable<K, V>> + 't> FullBucket<K, V, M> {
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

impl<'t, K, V, M: DerefMut<RawTable<K, V>> + 't> FullBucket<K, V, M> {
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
            Empty(..) => fail!("Expected full bucket")
        }
    }
}

impl<K, V, M: Deref<RawTable<K, V>>> GapThenFull<K, V, M> {
    #[inline]
    pub fn full(&self) -> &FullBucket<K, V, M> {
        &self.full
    }

    pub fn shift(mut self) -> Option<GapThenFull<K, V, M>> {
        unsafe {
            *self.gap.raw.hash = mem::replace(&mut *self.full.raw.hash, EMPTY_BUCKET);
            copy_nonoverlapping_memory(self.gap.raw.key, self.full.raw.key as *const K, 1);
            copy_nonoverlapping_memory(self.gap.raw.val, self.full.raw.val as *const V, 1);
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
/// # Failure
///
/// Fails if `target_alignment` is not a power of two.
fn round_up_to_next(unrounded: uint, target_alignment: uint) -> uint {
    assert!(is_power_of_two(target_alignment));
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
fn calculate_offsets(hashes_size: uint,
                     keys_size: uint, keys_align: uint,
                     vals_align: uint)
                     -> (uint, uint) {
    let keys_offset = round_up_to_next(hashes_size, keys_align);
    let end_of_keys = keys_offset + keys_size;

    let vals_offset = round_up_to_next(end_of_keys, vals_align);

    (keys_offset, vals_offset)
}

// Returns a tuple of (minimum required malloc alignment, hash_offset,
// array_size), from the start of a mallocated array.
fn calculate_allocation(hash_size: uint, hash_align: uint,
                        keys_size: uint, keys_align: uint,
                        vals_size: uint, vals_align: uint)
                        -> (uint, uint, uint) {
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
    unsafe fn new_uninitialized(capacity: uint) -> RawTable<K, V> {
        if capacity == 0 {
            return RawTable {
                size: 0,
                capacity: 0,
                hashes: 0 as *mut u64,
                marker: marker::CovariantType,
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
        let size_of_bucket = size_of::<u64>().checked_add(&size_of::<K>()).unwrap()
                                             .checked_add(&size_of::<V>()).unwrap();
        assert!(size >= capacity.checked_mul(&size_of_bucket)
                                .expect("capacity overflow"),
                "capacity overflow");

        let buffer = allocate(size, malloc_alignment);

        let hashes = buffer.offset(hash_offset as int) as *mut u64;

        RawTable {
            capacity: capacity,
            size:     0,
            hashes:   hashes,
            marker:   marker::CovariantType,
        }
    }

    fn first_bucket_raw(&self) -> RawBucket<K, V> {
        let hashes_size = self.capacity * size_of::<u64>();
        let keys_size = self.capacity * size_of::<K>();

        let buffer = self.hashes as *mut u8;
        let (keys_offset, vals_offset) = calculate_offsets(hashes_size,
                                                           keys_size, min_align_of::<K>(),
                                                           min_align_of::<V>());

        unsafe {
            RawBucket {
                hash: self.hashes,
                key:  buffer.offset(keys_offset as int) as *mut K,
                val:  buffer.offset(vals_offset as int) as *mut V
            }
        }
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    #[allow(experimental)]
    pub fn new(capacity: uint) -> RawTable<K, V> {
        unsafe {
            let ret = RawTable::new_uninitialized(capacity);
            zero_memory(ret.hashes, capacity);
            ret
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> uint {
        self.capacity
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> uint {
        self.size
    }

    fn raw_buckets(&self) -> RawBuckets<K, V> {
        RawBuckets {
            raw: self.first_bucket_raw(),
            hashes_end: unsafe {
                self.hashes.offset(self.capacity as int)
            }
        }
    }

    pub fn iter(&self) -> Entries<K, V> {
        Entries {
            iter: self.raw_buckets(),
            elems_left: self.size(),
        }
    }

    pub fn iter_mut(&mut self) -> MutEntries<K, V> {
        MutEntries {
            iter: self.raw_buckets(),
            elems_left: self.size(),
        }
    }

    pub fn into_iter(self) -> MoveEntries<K, V> {
        MoveEntries {
            iter: self.raw_buckets(),
            table: self,
        }
    }

    /// Returns an iterator that copies out each entry. Used while the table
    /// is being dropped.
    unsafe fn rev_move_buckets(&mut self) -> RevMoveBuckets<K, V> {
        let raw_bucket = self.first_bucket_raw();
        RevMoveBuckets {
            raw: raw_bucket.offset(self.capacity as int),
            hashes_end: raw_bucket.hash,
            elems_left: self.size
        }
    }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
struct RawBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64
}

impl<'a, K, V> Iterator<RawBucket<K, V>> for RawBuckets<'a, K, V> {
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
/// in an an inconsistent state and should only be used for dropping
/// the table's remaining entries. It's used in the implementation of Drop.
struct RevMoveBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64,
    elems_left: uint
}

impl<'a, K, V> Iterator<(K, V)> for RevMoveBuckets<'a, K, V> {
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
                        ptr::read(self.raw.key as *const K),
                        ptr::read(self.raw.val as *const V)
                    ));
                }
            }
        }
    }
}

/// Iterator over shared references to entries in a table.
pub struct Entries<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: uint,
}

/// Iterator over mutable references to entries in a table.
pub struct MutEntries<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: uint,
}

/// Iterator over the entries in a table, consuming the table.
pub struct MoveEntries<K, V> {
    table: RawTable<K, V>,
    iter: RawBuckets<'static, K, V>
}

impl<'a, K, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                (&*bucket.key,
                 &*bucket.val)
            }
        })
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.elems_left, Some(self.elems_left))
    }
}

impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                (&*bucket.key,
                 &mut *bucket.val)
            }
        })
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.elems_left, Some(self.elems_left))
    }
}

impl<K, V> Iterator<(SafeHash, K, V)> for MoveEntries<K, V> {
    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                (
                    SafeHash {
                        hash: *bucket.hash,
                    },
                    ptr::read(bucket.key as *const K),
                    ptr::read(bucket.val as *const V)
                )
            }
        })
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let size = self.table.size();
        (size, Some(size))
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
        if self.hashes.is_null() {
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
            deallocate(self.hashes as *mut u8, size, align);
            // Remember how everything was allocated out of one buffer
            // during initialization? We only need one call to free here.
        }
    }
}
